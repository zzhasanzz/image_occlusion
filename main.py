from extract_figures import extract_figures_from_pdf
from detect_labels import detect_labels
from visualize_boxes import visualize_boxes
from generate_flashcards import generate_flashcards
import sys
from warnings import filterwarnings


filterwarnings("ignore", category=UserWarning, module="torch")

if __name__ == "__main__":
    PDF_PATH = "input/anatomy_v2.pdf"
    FIGURES_DIR = "output/figures_v2"
    JSON_PATH = "output/merged_boxes_all.json"

    if len(sys.argv) >= 2:
        start_page = int(sys.argv[1])
        end_page = int(sys.argv[2])
    else:
        start_page = 17
        end_page = 31  # Default to first 10 pages if not specified

    # Step 1: Extract figures
    extract_figures_from_pdf(PDF_PATH, FIGURES_DIR, page_range=range(start_page,end_page+1))

    # Step 2: OCR + merge labels
    detect_labels(FIGURES_DIR, JSON_PATH)

    # Step 3: Visualize boxes
    visualize_boxes(FIGURES_DIR, JSON_PATH, "output/annotated")

    # Step 4: Generate flashcards
    generate_flashcards(FIGURES_DIR, JSON_PATH, "output/flashcards")
