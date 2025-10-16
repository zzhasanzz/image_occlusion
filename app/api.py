import os, shutil, tempfile, json
from pathlib import Path
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

# ==== Your pipeline imports ====
from extract_figures import extract_figures_from_pdf
from detect_labels import detect_labels
from visualize_boxes import visualize_boxes
from generate_flashcards import generate_flashcards
from rag_vlm import answer_from_image_and_query

# ================================================================
# 📘 APP CONFIGURATION
# ================================================================
app = FastAPI(title="Anatomy Tutor API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directories (absolute)
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures_v2"
DATA_DIR = BASE_DIR / "data"

JSON_PATH = OUTPUT_DIR / "merged_boxes_all.json"
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "index.pkl"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Static mount (so images are accessible to frontend)
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# ================================================================
# 🩺 HEALTH CHECK
# ================================================================
@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "2.0", "figures_dir": str(FIGURES_DIR)}

# ================================================================
# 📄 PROCESS PDF PIPELINE
# ================================================================
@app.post("/process_pdf")
async def process_pdf(file: UploadFile, start_page: int = Form(...), end_page: int = Form(...)):
    """
    Accept a PDF, extract figures, OCR labels, and generate flashcards.
    """
    tmp_dir = tempfile.mkdtemp()
    tmp_pdf_path = Path(tmp_dir) / file.filename
    with open(tmp_pdf_path, "wb") as f:
        f.write(await file.read())

    extract_figures_from_pdf(str(tmp_pdf_path), str(FIGURES_DIR), page_range=range(start_page - 1, end_page))
    detect_labels(str(FIGURES_DIR), str(JSON_PATH))
    visualize_boxes(str(FIGURES_DIR), str(JSON_PATH), str(OUTPUT_DIR / "annotated"))
    generate_flashcards(str(FIGURES_DIR), str(JSON_PATH), str(OUTPUT_DIR / "flashcards"))

    return {
        "status": "✅ success",
        "figures_path": str(FIGURES_DIR),
        "flashcards_path": str(OUTPUT_DIR / "flashcards"),
        "annotated_path": str(OUTPUT_DIR / "annotated"),
        "json_path": str(JSON_PATH),
        "message": f"Processed pages {start_page}–{end_page} from {file.filename}"
    }

# ================================================================
# 📦 DOWNLOAD RESULTS
# ================================================================
@app.get("/download/{folder}")
async def download_folder(folder: str):
    valid = ["figures_v2", "flashcards", "annotated"]
    if folder not in valid:
        return JSONResponse(content={"error": "Invalid folder"}, status_code=400)

    folder_path = OUTPUT_DIR / folder
    if not folder_path.exists():
        return JSONResponse(content={"error": "Folder not found"}, status_code=404)

    zip_path = folder_path.with_suffix(".zip")
    shutil.make_archive(str(folder_path), 'zip', str(folder_path))
    return FileResponse(str(zip_path), filename=zip_path.name)

# ================================================================
# 🖼️ PREVIEW IMAGES
# ================================================================
@app.get("/preview/{folder}")
async def preview_folder(folder: str):
    valid = ["figures_v2", "flashcards", "annotated"]
    if folder not in valid:
        return JSONResponse(content={"error": "Invalid folder"}, status_code=400)

    folder_path = OUTPUT_DIR / folder
    if not folder_path.exists():
        return JSONResponse(content={"error": "Folder not found"}, status_code=404)

    files = sorted([f"/output/{folder}/{fname}" for fname in os.listdir(folder_path)
                    if fname.lower().endswith((".png", ".jpg", ".jpeg"))])

    return {"count": len(files), "images": files}

# ================================================================
# 💬 CHAT MODULE (merged from chat_api)
# ================================================================
@app.get("/chat/list_images")
async def list_images():
    """
    Return all available figure images.
    """
    if not FIGURES_DIR.exists():
        return {"images": [], "message": f"⚠️ No folder found: {FIGURES_DIR}"}

    files = sorted([
        f for f in os.listdir(FIGURES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if not files:
        return {"images": [], "message": "⚠️ No extracted figures found. Please process a PDF first."}

    return {"count": len(files), "images": files}

@app.get("/chat/get_bounding_boxes")
async def get_bounding_boxes(image_name: str = None):
    """
    Return bounding boxes for images.
    If image_name is provided, return boxes for that specific image.
    If no image_name, return all bounding boxes.
    """
    if not JSON_PATH.exists():
        return JSONResponse(
            {"error": "Bounding boxes file not found. Please process a PDF first."}, 
            status_code=404
        )
    
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            all_boxes = json.load(f)
        
        # If specific image requested
        if image_name:
            if image_name in all_boxes:
                return {
                    "image_name": image_name,
                    "boxes": all_boxes[image_name],
                    "total_boxes": len(all_boxes[image_name])
                }
            else:
                # Check if image exists but has no boxes
                image_path = FIGURES_DIR / image_name
                if image_path.exists():
                    return {
                        "image_name": image_name,
                        "boxes": [],
                        "total_boxes": 0,
                        "message": "No bounding boxes found for this image"
                    }
                else:
                    return JSONResponse(
                        {"error": f"Image '{image_name}' not found"}, 
                        status_code=404
                    )
        
        # Return all boxes if no specific image requested
        return {
            "all_boxes": all_boxes,
            "total_images": len(all_boxes),
            "message": f"Found bounding boxes for {len(all_boxes)} images"
        }
        
    except json.JSONDecodeError as e:
        return JSONResponse(
            {"error": f"Invalid JSON in bounding boxes file: {str(e)}"}, 
            status_code=500
        )
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to load bounding boxes: {str(e)}"}, 
            status_code=500
        )

@app.get("/chat/get_all_bounding_boxes")
async def get_all_bounding_boxes():
    """
    Return all bounding boxes for all images.
    """
    return await get_bounding_boxes()

@app.post("/chat/chat_with_image")
async def chat_with_image(user_query: str = Form(...),
                          image_name: str = Form(...),
                          use_gemini: bool = Form(True)):
    """
    Chat with extracted figure — multimodal RAG reasoning.
    """
    image_path = FIGURES_DIR / image_name
    if not image_path.exists():
        return JSONResponse({"error": f"Image '{image_name}' not found."}, status_code=404)

    try:
        answer, sources, smart_q = answer_from_image_and_query(
            image_path=str(image_path),
            user_query=user_query,
            json_path=str(JSON_PATH),
            index_path=str(INDEX_PATH),
            meta_path=str(META_PATH),
            use_gemini=use_gemini
        )

        return {
            "status": "✅ success",
            "image_name": image_name,
            "query_used": smart_q,
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ================================================================
# 🔍 DEBUG ENDPOINTS (optional - for development)
# ================================================================
@app.get("/debug/bounding_boxes_status")
async def debug_bounding_boxes_status():
    """
    Debug endpoint to check bounding boxes file status.
    """
    status = {
        "json_path": str(JSON_PATH),
        "json_exists": JSON_PATH.exists(),
        "figures_dir": str(FIGURES_DIR),
        "figures_exists": FIGURES_DIR.exists(),
    }
    
    if JSON_PATH.exists():
        try:
            with open(JSON_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            status.update({
                "total_images_with_boxes": len(data),
                "sample_images": list(data.keys())[:3] if data else []
            })
        except Exception as e:
            status["json_error"] = str(e)
    
    if FIGURES_DIR.exists():
        image_files = [f for f in os.listdir(FIGURES_DIR) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        status.update({
            "total_images_in_folder": len(image_files),
            "sample_images_in_folder": image_files[:3] if image_files else []
        })
    
    return status