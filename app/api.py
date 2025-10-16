# app/api.py
import os, shutil, tempfile
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from extract_figures import extract_figures_from_pdf
from detect_labels import detect_labels
from visualize_boxes import visualize_boxes
from generate_flashcards import generate_flashcards
from starlette.middleware.cors import CORSMiddleware

app = FastAPI(title="Anatomy Tutor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "output"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures_v2")
JSON_PATH = os.path.join(OUTPUT_DIR, "merged_boxes_all.json")

@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok"})

@app.post("/process_pdf")
async def process_pdf(
    file: UploadFile,
    start_page: int = Form(...),
    end_page: int = Form(...)
):
    """
    Accept a PDF upload
    Run extraction pipeline (figures + OCR + flashcards)
    Return a success message and output folder info
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save uploaded file temporarily
    tmp_dir = tempfile.mkdtemp()
    tmp_pdf_path = os.path.join(tmp_dir, file.filename)
    with open(tmp_pdf_path, "wb") as f:
        f.write(await file.read())

    # Step 1: Extract figures
    extract_figures_from_pdf(tmp_pdf_path, FIGURES_DIR, page_range=range(start_page - 1, end_page))

    # Step 2: Detect labels + generate flashcards
    detect_labels(FIGURES_DIR, JSON_PATH)
    visualize_boxes(FIGURES_DIR, JSON_PATH, os.path.join(OUTPUT_DIR, "annotated"))
    generate_flashcards(FIGURES_DIR, JSON_PATH, os.path.join(OUTPUT_DIR, "flashcards"))

    # Return summary
    response = {
        "status": "✅ success",
        "figures_path": FIGURES_DIR,
        "flashcards_path": os.path.join(OUTPUT_DIR, "flashcards"),
        "annotated_path": os.path.join(OUTPUT_DIR, "annotated"),
        "json_path": JSON_PATH,
        "message": f"Processed pages {start_page}–{end_page} from {file.filename}"
    }
    return JSONResponse(content=response)

@app.get("/download/{folder}")
async def download_folder(folder: str):
    """
    Allow downloading zipped results (optional)
    """
    if folder not in ["figures_v2", "flashcards", "annotated"]:
        return JSONResponse(content={"error": "Invalid folder"}, status_code=400)

    folder_path = os.path.join(OUTPUT_DIR, folder)
    if not os.path.exists(folder_path):
        return JSONResponse(content={"error": "Folder not found"}, status_code=404)

    zip_path = f"{folder_path}.zip"
    shutil.make_archive(folder_path, 'zip', folder_path)
    return FileResponse(zip_path, filename=os.path.basename(zip_path))
