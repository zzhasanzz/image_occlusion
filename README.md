# Anatomy Flashcard Generator

A pipeline that extracts anatomical figures from PDFs, detects labels via OCR, and creates interactive flashcards.

### Steps
1. **Extract Figures** – Uses `Aryn/deformable-detr-DocLayNet` model.
2. **Detect Labels** – Uses EasyOCR with adaptive merging.
3. **Visualize** – Draws bounding boxes to verify.
4. **Generate Flashcards** – Creates masked “guess the label” images.

### Run
```bash
pip install -r requirements.txt
python main.py
````

Output:

* `output/figures_v2/` → extracted images
* `output/annotated/` → annotated bounding boxes
* `output/flashcards/` → generated flashcards

