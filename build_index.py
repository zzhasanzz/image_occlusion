import os, pickle, faiss, fitz, textwrap
from sentence_transformers import SentenceTransformer

PDF_PATH = "input/anatomy_v3.pdf"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
META_FILE  = os.path.join(DATA_DIR, "index.pkl")
CHUNK_SIZE = 800

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dimension = embedder.get_sentence_embedding_dimension()

print(f"[INFO] Reading PDF: {PDF_PATH}")
doc = fitz.open(PDF_PATH)
documents, metadata = [], []

for i, page in enumerate(doc):
    text = page.get_text("text").strip()
    if not text:
        continue
    chunks = [text[j:j+CHUNK_SIZE] for j in range(0, len(text), CHUNK_SIZE)]
    for chunk in chunks:
        documents.append(chunk)
        metadata.append(f"page {i+1}")
doc.close()

print(f"[INFO] {len(documents)} chunks → generating embeddings...")
embeddings = embedder.encode(documents, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, INDEX_FILE)
with open(META_FILE, "wb") as f:
    pickle.dump({"documents": documents, "metadata": metadata}, f)

print(f"✅ Index built with {index.ntotal} vectors.")
