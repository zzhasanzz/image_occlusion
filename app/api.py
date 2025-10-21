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

JSON_PATH = OUTPUT_DIR / "anatomy_v3_merged_boxes.json"
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
    Automatically includes the book name in all generated files/folders.
    """
    # ---------------- Prepare temporary file ----------------
    tmp_dir = tempfile.mkdtemp()
    tmp_pdf_path = Path(tmp_dir) / file.filename
    with open(tmp_pdf_path, "wb") as f:
        f.write(await file.read())

    # ---------------- Infer book name ----------------
    book_name = Path(file.filename).stem.replace(" ", "_")
    print(f"📚 Processing book: {book_name}")

    # ---------------- Create per-book output folders ----------------
    book_figures_dir = FIGURES_DIR / book_name
    book_flashcards_dir = OUTPUT_DIR / "flashcards" / book_name
    book_annotated_dir = OUTPUT_DIR / "annotated" / book_name
    book_json_path = OUTPUT_DIR / f"{book_name}_merged_boxes.json"

    book_figures_dir.mkdir(parents=True, exist_ok=True)
    book_flashcards_dir.mkdir(parents=True, exist_ok=True)
    book_annotated_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Run pipeline ----------------
    extract_figures_from_pdf(
        pdf_path=str(tmp_pdf_path),
        output_dir=str(book_figures_dir),
        page_range=range(start_page - 1, end_page)
    )

    detect_labels(str(book_figures_dir), str(book_json_path))
    visualize_boxes(str(book_figures_dir), str(book_json_path), str(book_annotated_dir))

    generate_flashcards(
        images_folder=str(book_figures_dir),
        json_path=str(book_json_path),
        output_folder=str(OUTPUT_DIR / "flashcards"),
        book_name=book_name
    )

    # ---------------- Return response ----------------
    return {
        "status": "✅ success",
        "book_name": book_name,
        "figures_path": str(book_figures_dir),
        "flashcards_path": str(book_flashcards_dir),
        "annotated_path": str(book_annotated_dir),
        "json_path": str(book_json_path),
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
async def preview_folder(folder: str, book_name: str | None = None):
    """
    Preview images from a folder (figures_v2, flashcards, annotated).
    Supports per-book subfolders like flashcards/{book_name}/.
    """
    valid = ["figures_v2", "flashcards", "annotated"]
    if folder not in valid:
        return JSONResponse(content={"error": "Invalid folder"}, status_code=400)

    base_folder = OUTPUT_DIR / folder

    # Check if folder exists
    if not base_folder.exists():
        return JSONResponse(content={"error": "Folder not found"}, status_code=404)

    # If a book name is provided, preview only that subfolder
    if book_name:
        book_folder = base_folder / book_name
        if not book_folder.exists():
            return JSONResponse(content={"error": f"Book '{book_name}' not found in {folder}"}, status_code=404)

        files = sorted([
            f"/output/{folder}/{book_name}/{fname}"
            for fname in os.listdir(book_folder)
            if fname.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        return {
            "book_name": book_name,
            "count": len(files),
            "images": files
        }

    # If no book_name provided, aggregate all books
    all_books = [
        d for d in os.listdir(base_folder)
        if (base_folder / d).is_dir()
    ]

    all_images = {}
    total_count = 0

    for bname in all_books:
        subfolder = base_folder / bname
        imgs = sorted([
            f"/output/{folder}/{bname}/{fname}"
            for fname in os.listdir(subfolder)
            if fname.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if imgs:
            all_images[bname] = imgs
            total_count += len(imgs)

    return {
        "total_books": len(all_books),
        "total_images": total_count,
        "books": all_books,
        "images_by_book": all_images
    }


# ================================================================
# 💬 CHAT MODULE (merged from chat_api)
# ================================================================
@app.get("/chat/list_images")
async def list_images(book_name: str | None = None):
    """
    Return all available figure images.
    If book_name is provided, return figures only for that book.
    """
    if not FIGURES_DIR.exists():
        return {"images": [], "message": f"⚠️ No folder found: {FIGURES_DIR}"}

    # If specific book requested
    if book_name:
        book_folder = FIGURES_DIR / book_name
        if not book_folder.exists():
            return {"images": [], "message": f"⚠️ No figures found for book '{book_name}'."}

        files = sorted([
            f"/output/figures_v2/{book_name}/{fname}"
            for fname in os.listdir(book_folder)
            if fname.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        if not files:
            return {"images": [], "message": f"⚠️ No images in book '{book_name}' folder."}

        return {"book_name": book_name, "count": len(files), "images": files}

    # If no book name — aggregate all books
    all_books = [
        d for d in os.listdir(FIGURES_DIR)
        if (FIGURES_DIR / d).is_dir()
    ]

    all_images = {}
    total_count = 0

    for bname in all_books:
        subfolder = FIGURES_DIR / bname
        imgs = sorted([
            f"/output/figures_v2/{bname}/{fname}"
            for fname in os.listdir(subfolder)
            if fname.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if imgs:
            all_images[bname] = imgs
            total_count += len(imgs)

    if total_count == 0:
        return {"images": [], "message": "⚠️ No extracted figures found. Please process a PDF first."}

    return {
        "total_books": len(all_books),
        "total_images": total_count,
        "books": all_books,
        "images_by_book": all_images
    }


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

# ================================================================
# 🧠 FIGURE CATEGORIZATION ENDPOINT (MiniLM-based)
# ================================================================
import pickle, torch, re
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from fastapi.responses import JSONResponse

@app.post("/categorize")
async def categorize_figures():
    """
    Categorize each anatomical figure into a single best anatomical category.
    Uses rule-based label matching first, then MiniLM semantic similarity fallback.
    """

    MERGED_FILE = OUTPUT_DIR / "anatomy_v3_merged_boxes.json"
    PKL_FILE = DATA_DIR / "classified_anatomy_mammoth.pkl"
    OUTPUT_FILE = OUTPUT_DIR / "figure_category_map.json"

    # ---------------- Validate Files ----------------
    if not MERGED_FILE.exists():
        return JSONResponse({"error": f"Missing file: {MERGED_FILE}"}, status_code=404)
    if not PKL_FILE.exists():
        return JSONResponse({"error": f"Missing file: {PKL_FILE}"}, status_code=404)

    # ---------------- Load Inputs ----------------
    print(f"[INFO] Loading merged boxes from: {MERGED_FILE}")
    with open(MERGED_FILE, "r", encoding="utf-8") as f:
        merged_boxes = json.load(f)

    print(f"[INFO] Loading classification dictionary from: {PKL_FILE}")
    with open(PKL_FILE, "rb") as f:
        classified_terms = pickle.load(f)

    print(f"[INFO] Loaded {len(classified_terms)} anatomical categories")

    # ---------------- Load Embedding Model ----------------
    print("[INFO] Loading MiniLM model (sentence-transformers/all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Precompute category centroids
    category_embeddings = {}
    for cat, terms in classified_terms.items():
        texts = [t.replace("_", " ") for t in terms[:300]]  # limit for speed
        with torch.no_grad():
            embs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        centroid = torch.mean(embs, dim=0)
        category_embeddings[cat] = centroid

    # ---------------- Helper Function ----------------
    def normalize(text):
        text = text.lower().strip()
        text = re.sub(r"[^a-z\s]", " ", text)
        return text

    # ---------------- Categorization Logic ----------------
    figure_to_category = {}

    for fig, entries in merged_boxes.items():
        labels = [normalize(obj["text"]) for obj in entries]
        label_text = " ".join(labels)

        counter = Counter()
        for label in labels:
            for cat, terms in classified_terms.items():
                for term in terms:
                    if term in label:
                        counter[cat] += 1

        # Rule-based match
        if counter:
            best_cat, _ = counter.most_common(1)[0]
            figure_to_category[fig] = best_cat
            continue

        # Fallback: semantic similarity
        with torch.no_grad():
            query_emb = model.encode(label_text, convert_to_tensor=True, normalize_embeddings=True)
        sims = {cat: float(util.cos_sim(query_emb, centroid)) for cat, centroid in category_embeddings.items()}
        best_cat = max(sims.items(), key=lambda x: x[1])[0]
        figure_to_category[fig] = best_cat

    # ---------------- Save & Return ----------------
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(figure_to_category, f, indent=2, ensure_ascii=False)

    return {
        "status": "✅ success",
        "message": f"Categorized {len(figure_to_category)} figures",
        "output_path": str(OUTPUT_FILE)
    }


@app.get("/get_categorization_summary")
async def get_categorization_summary():
    """
    Return the categorization summary with figures organized by category
    """
    OUTPUT_FILE = OUTPUT_DIR / "figure_category_map.json"
    
    if not OUTPUT_FILE.exists():
        return JSONResponse({"error": "Categorization file not found"}, status_code=404)
    
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            figure_to_category = json.load(f)
        
        # Organize figures by category
        categories = {}
        for figure, category in figure_to_category.items():
            if category not in categories:
                categories[category] = []
            categories[category].append(figure)
        
        return {
            "total_figures": len(figure_to_category),
            "categories": categories
        }
    except Exception as e:
        return JSONResponse({"error": f"Failed to load categorization: {str(e)}"}, status_code=500)


import datetime, random, json, os
from pathlib import Path
from fastapi import Form
from fastapi.responses import JSONResponse

# ================================================================
# 🧠 FLASHCARD REVIEW ENDPOINTS (Spaced Repetition)
# ================================================================
FLASHCARDS_BASE = OUTPUT_DIR / "flashcards"
INTERVALS = {"easy": 14, "normal": 7, "hard": 1, "repeat": 0}


def load_flashcards_state(book_name: str):
    """Load or initialize a flashcard JSON state for a given book."""
    book_folder = FLASHCARDS_BASE / book_name
    state_file = book_folder / f"{book_name}_flashcards.json"

    if not book_folder.exists():
        raise FileNotFoundError(f"⚠️ Flashcards folder not found: {book_folder}")

    if not state_file.exists():
        # Initialize automatically if missing
        files = sorted([f for f in os.listdir(book_folder) if f.endswith(".png")])
        q_files = [f for f in files if "_answer" not in f]

        pairs = {}
        for q_name in q_files:
            base = q_name.replace(".png", "")
            a_name = f"{base}_answer.png"
            if not (book_folder / a_name).exists():
                continue

            parts = base.split("_")
            page, fig = None, None
            for p in parts:
                if p.startswith("page"): page = p[4:]
                if p.startswith("fig"): fig = p[3:]

            pairs[q_name] = {
                "question": f"/output/flashcards/{book_name}/{q_name}",
                "answer": f"/output/flashcards/{book_name}/{a_name}",
                "page": page,
                "figure": fig,
                "difficulty": None,
                "next_repeat_date": datetime.date.today().isoformat()
            }

        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        return pairs, state_file

    with open(state_file, "r", encoding="utf-8") as f:
        cards = json.load(f)
    return cards, state_file


def get_due_cards(cards):
    today = datetime.date.today().isoformat()
    return {k: v for k, v in cards.items() if v["next_repeat_date"] <= today}


# ================================================================
# 📅 GET /flashcards/due/{book_name}
# ================================================================
@app.get("/flashcards/due/{book_name}")
async def get_due_flashcards(book_name: str):
    """Return flashcards due for review today for the given book."""
    try:
        cards, _ = load_flashcards_state(book_name)
        due = get_due_cards(cards)
        if not due:
            return {"book_name": book_name, "due_count": 0, "message": "🎉 No flashcards due today!"}

        keys = list(due.keys())
        random.shuffle(keys)
        due_cards = [due[k] for k in keys]

        return {
            "book_name": book_name,
            "due_count": len(due_cards),
            "due_cards": due_cards
        }
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": f"Failed to load flashcards: {e}"}, status_code=500)


# ================================================================
# 🧩 POST /flashcards/feedback
# ================================================================
@app.post("/flashcards/feedback")
async def update_flashcard_feedback(
    book_name: str = Form(...),
    question_file: str = Form(...),
    feedback: str = Form(...)
):
    """
    Update flashcard difficulty and next_repeat_date
    based on user feedback (easy / normal / hard / repeat).
    """
    try:
        book_folder = FLASHCARDS_BASE / book_name
        state_file = book_folder / f"{book_name}_flashcards.json"

        if not state_file.exists():
            return JSONResponse({"error": f"State file not found for book '{book_name}'"}, status_code=404)

        with open(state_file, "r", encoding="utf-8") as f:
            cards = json.load(f)

        if question_file not in cards:
            return JSONResponse({"error": f"Flashcard '{question_file}' not found."}, status_code=404)

        if feedback not in INTERVALS:
            return JSONResponse({"error": f"Invalid feedback: {feedback}"}, status_code=400)

        delta_days = INTERVALS[feedback]
        next_date = datetime.date.today() + datetime.timedelta(days=delta_days)
        cards[question_file]["difficulty"] = feedback
        cards[question_file]["next_repeat_date"] = next_date.isoformat()

        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(cards, f, indent=2, ensure_ascii=False)

        return {
            "status": "✅ updated",
            "book_name": book_name,
            "question": question_file,
            "difficulty": feedback,
            "next_repeat_date": next_date.isoformat()
        }

    except Exception as e:
        return JSONResponse({"error": f"Failed to update feedback: {e}"}, status_code=500)
