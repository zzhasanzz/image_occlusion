import json, re, torch, pickle
from collections import Counter
from sentence_transformers import SentenceTransformer, util

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
MERGED_FILE = "output/anatomy_v3_merged_boxes_all.json"             # from your OCR output
PKL_FILE = "data/classified_anatomy_mammoth.pkl"       # built using build_anatomy_pkl.py
OUTPUT_FILE = "output/figure_category_map.json"          # final one-category-per-figure map

# ------------------------------------------------------------
# 1️⃣ Load Inputs
# ------------------------------------------------------------
print(f"[INFO] Loading: {MERGED_FILE}")
with open(MERGED_FILE, "r", encoding="utf-8") as f:
    merged_boxes = json.load(f)

print(f"[INFO] Loading classification dictionary from: {PKL_FILE}")
with open(PKL_FILE, "rb") as f:
    classified_terms = pickle.load(f)

print(f"[INFO] Loaded {len(classified_terms)} anatomical categories")

# ------------------------------------------------------------
# 2️⃣ Load Embedding Model
# ------------------------------------------------------------
print("[INFO] Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Precompute centroid embeddings per category
category_embeddings = {}
for cat, terms in classified_terms.items():
    texts = [t.replace("_", " ") for t in terms[:300]]  # limit for speed
    with torch.no_grad():
        embs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    centroid = torch.mean(embs, dim=0)
    category_embeddings[cat] = centroid

print(f"[INFO] Prepared {len(category_embeddings)} category centroids.")

# ------------------------------------------------------------
# 3️⃣ Normalize Label Text
# ------------------------------------------------------------
def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text

# ------------------------------------------------------------
# 4️⃣ Assign Each Figure a Single Best Category
# ------------------------------------------------------------
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

    if counter:
        best_cat, _ = counter.most_common(1)[0]
        figure_to_category[fig] = best_cat
        continue

    with torch.no_grad():
        query_emb = model.encode(label_text, convert_to_tensor=True, normalize_embeddings=True)
    sims = {cat: float(util.cos_sim(query_emb, centroid)) for cat, centroid in category_embeddings.items()}
    best_cat = max(sims.items(), key=lambda x: x[1])[0]
    figure_to_category[fig] = best_cat

# ------------------------------------------------------------
# 5️⃣ Save Results
# ------------------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(figure_to_category, f, indent=2, ensure_ascii=False)

print(f"[✅] Done! Saved figure → category map to {OUTPUT_FILE}")
