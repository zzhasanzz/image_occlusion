import cv2, os, json, re, numpy as np

# ─────────────────────────────────────────────────────────────
# 🔹 Helper: Detect invalid or ignorable labels
# ─────────────────────────────────────────────────────────────
def is_invalid_label(text: str) -> bool:
    """
    Returns True if the label is:
      - too short (<3 chars)
      - purely numeric or coordinate-like
      - figure caption (starts with 'fig' or 'figure')
    """
    t = text.strip()
    if len(t) < 3:
        return True

    # numeric patterns like: 1, L, -55 ~70, +30, etc.
    if re.fullmatch(r"[\+\-]?\d+(\.\d+)?(?:\s*[~–\-]\s*[\+\-]?\d+(\.\d+)?)?", t):
        return True

    # text like: Figure 2.4, Fig. 3, Fig 2A etc.
    if re.match(r"^(fig(ure)?\.?\s*\d+[a-zA-Z\-: ]*)", t, flags=re.IGNORECASE):
        return True

    # isolated letters or symbols
    if re.fullmatch(r"[A-Za-z]", t):
        return True

    return False


# ─────────────────────────────────────────────────────────────
# 🔹 Solid Color Overlay Function
# ─────────────────────────────────────────────────────────────
def apply_flashcard_mask(img, boxes, highlight_idx, reveal=False):
    """
    If reveal=False → draw solid glass-colored masks + dark blue border (question)
    If reveal=True  → mask everything except the highlighted box (answer)
    """
    overlay = img.copy()
    
    # Colors for solid masks and borders
    glass_color = (200, 220, 240)  # Light blue glass color (BGR)
    dark_blue = (139, 0, 0)  # Dark blue for borders (BGR: 139, 0, 0)
    highlight_border = (255, 191, 0)  # Light blue for highlighted border (BGR: 255, 191, 0)
    
    if not reveal:
        # QUESTION MODE - Apply solid color to all boxes
        for idx, b in enumerate(boxes):
            (x1, y1, x2, y2) = b["bbox"]
            
            # Apply SOLID glass color to completely hide underlying text
            cv2.rectangle(overlay, (x1, y1), (x2, y2), glass_color, -1)
            
            # Add dark blue border to all boxes
            border_color = highlight_border if idx == highlight_idx else dark_blue
            border_thickness = 3 if idx == highlight_idx else 1
            cv2.rectangle(overlay, (x1, y1), (x2, y2), border_color, border_thickness)
            
            # Add adaptive question marks to the highlighted box only
            if idx == highlight_idx:
                # Calculate font scale based on box size
                box_width = x2 - x1
                box_height = y2 - y1
                
                # Adaptive font scale based on box dimensions
                font_scale = max(0.6, min(box_width / 180, box_height / 60))
                thickness = max(1, int(font_scale * 2))
                
                # Add multiple question marks for better visibility
                question_text = "??Guess??"
                text_size = cv2.getTextSize(question_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                # Center the text in the box
                tx = x1 + (box_width - text_size[0]) // 2
                ty = y1 + (box_height + text_size[1]) // 2
                
                # Add dark blue text with white outline for maximum contrast
                # White outline
                cv2.putText(
                    overlay, question_text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness + 2, cv2.LINE_AA
                )
                # Dark blue text
                cv2.putText(
                    overlay, question_text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (220, 20, 60), thickness, cv2.LINE_AA
                )

    else:
        # ANSWER MODE - Apply solid color to all boxes EXCEPT the highlighted one
        for idx, b in enumerate(boxes):
            (x1, y1, x2, y2) = b["bbox"]
            
            if idx != highlight_idx:
                # Apply SOLID glass color to completely hide underlying text
                cv2.rectangle(overlay, (x1, y1), (x2, y2), glass_color, -1)
            
            # Add borders to all boxes
            border_color = highlight_border if idx == highlight_idx else dark_blue
            border_thickness = 2 if idx == highlight_idx else 1
            cv2.rectangle(overlay, (x1, y1), (x2, y2), border_color, border_thickness)

    return overlay


# ─────────────────────────────────────────────────────────────
# 🔹 Main Generator
# ─────────────────────────────────────────────────────────────
import os, json, cv2

def generate_flashcards(images_folder, json_path, output_folder="flashcards", book_name="Unknown_Book"):
    """
    Generate question/answer flashcards for each figure.
    Prevents book name duplication in filenames.
    """
    os.makedirs(output_folder, exist_ok=True)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for img_idx, (fname, boxes) in enumerate(list(data.items())):
        img_path = os.path.join(images_folder, fname)
        img = cv2.imread(img_path)
        if img is None or len(boxes) == 0:
            print(f"⚠️ Skipped: {fname}")
            continue

        # Filter invalid boxes
        valid_boxes = [b for b in boxes if not is_invalid_label(b.get("text", ""))]
        if not valid_boxes:
            print(f"⚠️ No valid labels found for {fname}, skipping.")
            continue

        print(f"\n🧩 Generating flashcards for: {fname} ({len(valid_boxes)} valid labels)")

        # Create subfolder for this book
        book_folder = os.path.join(output_folder, book_name)
        os.makedirs(book_folder, exist_ok=True)

        # ✅ Remove book name prefix if already in filename
        base_name = os.path.splitext(fname)[0]
        if base_name.startswith(book_name):
            base_name = base_name[len(book_name):].lstrip("_")

        for i, b in enumerate(valid_boxes):
            # Question (masked)
            q_img = apply_flashcard_mask(img, valid_boxes, i, reveal=False)
            q_name = f"{book_name}_{base_name}_q{i+1}.png"
            cv2.imwrite(os.path.join(book_folder, q_name), q_img)

            # Answer (revealed)
            a_img = apply_flashcard_mask(img, valid_boxes, i, reveal=True)
            a_name = f"{book_name}_{base_name}_q{i+1}_answer.png"
            cv2.imwrite(os.path.join(book_folder, a_name), a_img)

            print(f"✅ Saved: {q_name} & {a_name}")

    print(f"\n🎯 Done! Flashcards saved in '{output_folder}/{book_name}/'")

