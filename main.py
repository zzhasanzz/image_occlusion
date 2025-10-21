from extract_figures import extract_figures_from_pdf
from detect_labels import detect_labels
from visualize_boxes import visualize_boxes
from generate_flashcards import generate_flashcards
from rag_vlm import answer_from_image_and_query
import sys, textwrap, os
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning, module="torch")

if __name__ == "__main__":
    # ------------------------------------------------------------
    # 🧩 Paths and Config
    # ------------------------------------------------------------
    PDF_PATH = "input/anatomy_v3.pdf"     # uploaded book
    FIGURES_DIR = "output/figures_v2"
    JSON_PATH = "output/merged_boxes_all.json"
    DATA_DIR = "data"

    # Auto-extract book name from PDF filename
    book_name = os.path.splitext(os.path.basename(PDF_PATH))[0].replace(" ", "_")

    # Page range (default or via CLI args)
    start_page, end_page = (17, 31)
    if len(sys.argv) >= 3:
        start_page, end_page = int(sys.argv[1]), int(sys.argv[2])

    print(f"\n📚 Book: {book_name}")
    print(f"📖 Pages: {start_page}–{end_page}")

    # ------------------------------------------------------------
    # 1️⃣ Extract Figures
    # ------------------------------------------------------------
    extract_figures_from_pdf(
        pdf_path=PDF_PATH,
        output_dir=FIGURES_DIR,
        page_range=range(start_page - 1, end_page)
    )

    # ------------------------------------------------------------
    # 2️⃣ Detect Labels
    # ------------------------------------------------------------
    detect_labels(FIGURES_DIR, JSON_PATH)

    # ------------------------------------------------------------
    # 3️⃣ Visualize Boxes
    # ------------------------------------------------------------
    visualize_boxes(FIGURES_DIR, JSON_PATH, "output/annotated")

    # ------------------------------------------------------------
    # 4️⃣ Generate Flashcards (with book name)
    # ------------------------------------------------------------
    generate_flashcards(
        images_folder=FIGURES_DIR,
        json_path=JSON_PATH,
        output_folder="output/flashcards",
        book_name=book_name
    )

    # ------------------------------------------------------------
    # 5️⃣ Interactive Tutor (RAG + Vision-Language Model)
    # ------------------------------------------------------------
    print("\n=== 🔍 Interactive Tutor Mode ===")
    image_path = input("Enter image path (e.g., output/figures_v2/BD_Chaurasia_page031_fig01.png): ").strip()
    user_query = input("Ask a question about this figure: ").strip()

    answer, sources, smart_q = answer_from_image_and_query(
        image_path=image_path,
        user_query=user_query,
        json_path=JSON_PATH,
        index_path=f"{DATA_DIR}/faiss.index",
        meta_path=f"{DATA_DIR}/index.pkl",
        use_gemini=True
    )

    print("\n🧠 Smart Query:", smart_q)
    print("\n📚 Retrieved Context:\n", sources)
    print("\n💬 Tutor Answer:\n", textwrap.fill(answer, width=100))
