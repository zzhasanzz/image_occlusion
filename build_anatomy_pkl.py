import json
import pickle
import os

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
JSON_PATH = "data/classified_anatomy_mammoth.json"
OUTPUT_PKL = "data/classified_anatomy_mammoth.pkl"

# ------------------------------------------------------------
# 1️⃣ Load the JSON
# ------------------------------------------------------------
if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"❌ Could not find {JSON_PATH}")

print(f"[INFO] Loading classification JSON from: {JSON_PATH}")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"[INFO] Loaded {len(data)} categories")

# ------------------------------------------------------------
# 2️⃣ Save as Pickle for fast reloading
# ------------------------------------------------------------
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(data, f)

print(f"[✅] Saved pickle to: {OUTPUT_PKL}")
