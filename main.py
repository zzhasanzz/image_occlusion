from extract_figures import extract_figures_from_pdf
from detect_labels import detect_labels
from visualize_boxes import visualize_boxes
from generate_flashcards import generate_flashcards
from rag_vlm import answer_from_image_and_query
import sys, textwrap
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning, module="torch")

if __name__ == "__main__":
    PDF_PATH = "input/anatomy_v3.pdf"
    FIGURES_DIR = "output/figures_v2"
    JSON_PATH = "output/merged_boxes_all.json"
    DATA_DIR = "data"

    start_page, end_page = (17, 31)
    if len(sys.argv) >= 3:
        start_page, end_page = int(sys.argv[1]), int(sys.argv[2])

    # Step 1–4: existing pipeline
    extract_figures_from_pdf(PDF_PATH, FIGURES_DIR, page_range=range(start_page, end_page + 1))
    detect_labels(FIGURES_DIR, JSON_PATH)
    visualize_boxes(FIGURES_DIR, JSON_PATH, "output/annotated")
    generate_flashcards(FIGURES_DIR, JSON_PATH, "output/flashcards")

    # Step 5: interactive retrieval + VLM demo
    print("\n=== 🔍 Interactive Tutor Mode ===")
    image_path = input("Enter image path (e.g., output/figures_v2/page031_fig01.png): ").strip()
    user_query = input("Ask a question about this figure: ").strip()

    answer, sources, smart_q = answer_from_image_and_query(
        image_path=image_path,
        user_query=user_query,
        json_path=JSON_PATH,
        index_path=f"{DATA_DIR}/faiss.index",
        meta_path=f"{DATA_DIR}/index.pkl",
    )

    print("\n🧠 Smart query:", smart_q)
    print("\n📚 Retrieved context:\n", sources)
    print("\n💬 Tutor Answer:\n", textwrap.fill(answer, width=100))
