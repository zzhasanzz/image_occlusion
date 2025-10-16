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
# 🔹 Overlay Function
# ─────────────────────────────────────────────────────────────
def apply_flashcard_mask(img, boxes, highlight_idx, reveal=False):
    """
    If reveal=False → draw orange masks + crimson border (question)
    If reveal=True  → mask everything except the highlighted box (answer)
    """
    overlay = img.copy()
    dark_orange = (0, 110, 220)     # non-question fill
    highlight_fill = (0, 180, 255)  # question fill color
    crimson = (30, 20, 180)         # border color
    alpha = 1.0

    # fill all boxes with dark orange
    for b in boxes:
        (x1, y1, x2, y2) = b["bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), dark_orange, -1)

    (x1, y1, x2, y2) = boxes[highlight_idx]["bbox"]
    text = boxes[highlight_idx].get("text", "")

    if not reveal:
        # QUESTION MODE — highlight one region
        cv2.rectangle(overlay, (x1, y1), (x2, y2), highlight_fill, -1)
        blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        cv2.rectangle(blended, (x1, y1), (x2, y2), crimson, 2)

        # add adaptive question marks
        font_scale = max(0.5, (x2 - x1) / 150)
        thickness = 2
        text_size = cv2.getTextSize("???", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        tx = x1 + (x2 - x1 - text_size[0]) // 2
        ty = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(
            blended, "???", (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), thickness, cv2.LINE_AA
        )

    else:
        # ANSWER MODE — unmask the correct label region (reveal original)
        blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        blended[y1:y2, x1:x2] = img[y1:y2, x1:x2]  # restore original label region
        cv2.rectangle(blended, (x1, y1), (x2, y2), crimson, 2)

    return blended


# ─────────────────────────────────────────────────────────────
# 🔹 Main Generator
# ─────────────────────────────────────────────────────────────
def generate_flashcards(images_folder, json_path, output_folder="flashcards"):
    os.makedirs(output_folder, exist_ok=True)
    with open(json_path, "r") as f:
        data = json.load(f)

    for img_idx, (fname, boxes) in enumerate(list(data.items())):
        img_path = os.path.join(images_folder, fname)
        img = cv2.imread(img_path)
        if img is None or len(boxes) == 0:
            print(f"⚠️ Skipped: {fname}")
            continue

        # filter out invalid boxes
        valid_boxes = [b for b in boxes if not is_invalid_label(b.get("text", ""))]
        if not valid_boxes:
            print(f"⚠️ No valid labels found for {fname}, skipping.")
            continue

        print(f"\n🧩 Generating flashcards for: {fname} ({len(valid_boxes)} valid labels)")

        for i, b in enumerate(valid_boxes):
            # Question (masked)
            q_img = apply_flashcard_mask(img, valid_boxes, i, reveal=False)
            q_name = f"{os.path.splitext(fname)[0]}_q{i+1}.png"
            cv2.imwrite(os.path.join(output_folder, q_name), q_img)

            # Answer (revealed)
            a_img = apply_flashcard_mask(img, valid_boxes, i, reveal=True)
            a_name = f"{os.path.splitext(fname)[0]}_q{i+1}_answer.png"
            cv2.imwrite(os.path.join(output_folder, a_name), a_img)

            print(f"✅ Saved: {q_name} & {a_name}")

    print(f"\n🎯 Done! All flashcards and answers saved in '{output_folder}/'")
