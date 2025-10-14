import cv2, os, json, numpy as np

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

    if not reveal:
        # QUESTION MODE — highlight one region
        cv2.rectangle(overlay, (x1, y1), (x2, y2), highlight_fill, -1)
        blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        cv2.rectangle(blended, (x1, y1), (x2, y2), crimson, 2)
    else:
        # ANSWER MODE — unmask the correct label region (reveal original)
        blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        blended[y1:y2, x1:x2] = img[y1:y2, x1:x2]  # restore original label region
        cv2.rectangle(blended, (x1, y1), (x2, y2), crimson, 2)

    return blended


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

        print(f"\n🧩 Generating flashcards for: {fname} ({len(boxes)} labels)")

        for i in range(len(boxes)):
            # Question (highlight filled)
            q_img = apply_flashcard_mask(img, boxes, i, reveal=False)
            q_name = f"{os.path.splitext(fname)[0]}_q{i+1}.png"
            cv2.imwrite(os.path.join(output_folder, q_name), q_img)

            # Answer (revealed label)
            a_img = apply_flashcard_mask(img, boxes, i, reveal=True)
            a_name = f"{os.path.splitext(fname)[0]}_q{i+1}_answer.png"
            cv2.imwrite(os.path.join(output_folder, a_name), a_img)

            print(f"✅ Saved: {q_name} & {a_name}")

    print(f"\n🎯 Done! All flashcards and answers saved in '{output_folder}/'")
