import os, json, random, datetime, cv2
from pathlib import Path

# ============================================================
# 🧠 CONFIGURATION
# ============================================================
BOOK_NAME = "anatomy_v3"  # can be parameterized
FLASHCARDS_DIR = Path("output/flashcards") / BOOK_NAME
STATE_JSON = FLASHCARDS_DIR / f"{BOOK_NAME}_flashcards.json"

INTERVALS = {
    "easy": 14,
    "normal": 7,
    "hard": 1,
    "repeat": 0
}

# ============================================================
# 🧩 1️⃣ Initialize JSON (if not exists)
# ============================================================
def initialize_flashcards(book_name: str):
    """
    Create a JSON file storing all flashcards and their review schedule
    under that book's folder in flashcards/.
    """
    book_folder = Path("output/flashcards") / book_name
    if not book_folder.exists():
        raise FileNotFoundError(f"⚠️ Folder not found: {book_folder}")

    pairs = {}
    files = sorted([f for f in os.listdir(book_folder) if f.endswith(".png")])
    q_files = [f for f in files if "_answer" not in f]

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

    with open(STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"✅ Initialized flashcard state: {STATE_JSON}")
    return pairs


# ============================================================
# 🧠 2️⃣ Load Flashcards
# ============================================================
def load_flashcards():
    if not STATE_JSON.exists():
        print("⚠️ No JSON state file found, initializing...")
        return initialize_flashcards(BOOK_NAME)
    with open(STATE_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 🗓️ 3️⃣ Filter by Due Date
# ============================================================
def get_due_flashcards(cards):
    today = datetime.date.today().isoformat()
    due = {k: v for k, v in cards.items() if v["next_repeat_date"] <= today}
    return due


# ============================================================
# 🖼️ 4️⃣ Display Function
# ============================================================
def display_flashcard(image_path):
    img_path = image_path
    if image_path.startswith("/output/"):
        img_path = Path("output") / Path(image_path).relative_to("/output")
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️ Failed to open image: {img_path}")
        return
    cv2.imshow("Flashcard", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# 🔁 5️⃣ Review Session
# ============================================================
def review_session():
    cards = load_flashcards()
    due_cards = get_due_flashcards(cards)

    if not due_cards:
        print("🎉 No flashcards due today — great job!")
        return

    print(f"🧩 {len(due_cards)} flashcards to review for '{BOOK_NAME}' today.\n")
    keys = list(due_cards.keys())
    random.shuffle(keys)

    for key in keys:
        card = cards[key]
        print(f"\n📖 Page {card['page']} | Figure {card['figure'] or '?'}")

        print(f"❓ Question: {card['question']}")
        display_flashcard(card["question"])

        input("Press [Enter] to reveal answer...")
        display_flashcard(card["answer"])

        while True:
            feedback = input("How was it? (easy / normal / hard / repeat): ").strip().lower()
            if feedback in INTERVALS:
                break
            print("⚠️ Invalid choice. Try again.")

        delta_days = INTERVALS[feedback]
        next_date = datetime.date.today() + datetime.timedelta(days=delta_days)
        cards[key]["difficulty"] = feedback
        cards[key]["next_repeat_date"] = next_date.isoformat()

        print(f"✅ Next review on {next_date}")

        # Save incremental progress
        with open(STATE_JSON, "w", encoding="utf-8") as f:
            json.dump(cards, f, indent=2, ensure_ascii=False)

    print("\n🎯 Session complete! Progress saved.")


# ============================================================
# 🚀 MAIN
# ============================================================
if __name__ == "__main__":
    print(f"\n=== 🧠 Flashcard Review for '{BOOK_NAME}' ===")
    review_session()
