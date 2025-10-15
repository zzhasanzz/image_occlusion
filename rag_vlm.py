import os, json, pickle, faiss, numpy as np, textwrap, torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_NEW_TOKENS = 512
TOP_K = 3

def load_index(index_path, meta_path):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        data = pickle.load(f)
    return index, data["documents"], data["metadata"]

def load_embedder(model_name=EMBED_MODEL):
    return SentenceTransformer(model_name)

def retrieve_context(index, embedder, documents, metadata, query, top_k=TOP_K):
    q = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q, dtype=np.float32), top_k)
    results = []
    for rank, idx in enumerate(I[0]):
        if 0 <= idx < len(documents):
            results.append({
                "rank": rank + 1,
                "text": documents[idx],
                "source": metadata[idx],
                "distance": float(D[0][rank])
            })
    return results

def build_retrieval_query_from_json(image_name, user_query, json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    labels, caption = [], None
    for item in data.get(image_name, []):
        t = (item.get("text") or "").strip()
        if not t: 
            continue
        if t.lower().startswith("fig"):
            caption = t
        else:
            labels.append(t)
    terms = ", ".join(labels[:6])
    if caption and terms:
        return f"{user_query} (related to {caption}, {terms})"
    if caption:
        return f"{user_query} (related to {caption})"
    if terms:
        return f"{user_query} (related to {terms})"
    return user_query

def make_prompt(context, user_query):
    return f"""
You are an experienced anatomy tutor.
Use the textbook excerpts to answer accurately and tie your answer to the visible anatomy in the image.
Write a clear, instructional explanation (6–8 sentences).

Textbook context:
{context}

Question:
{user_query}

Answer:
""".strip()

def load_qwen(model_id="Qwen/Qwen2-VL-2B-Instruct", attn_impl="sdpa"):
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=attn_impl,
        device_map="auto"
    )
    return proc, model

def generate_answer(img, prompt, processor, model):
    # Try legacy utils first (official Qwen docs), else use HF-only path
    try:
        from qwen_vl_utils import process_vision_info
        messages = [{"role":"user","content":[
            {"type":"image","image":img},
            {"type":"text","text":prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7, top_p=0.9)
        resp = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return resp
    except Exception:
        # New HF path (no qwen_vl_utils)
        messages = [{"role":"user","content":[
            {"type":"image","image":img},
            {"type":"text","text":prompt},
        ]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7, top_p=0.9)
        return processor.batch_decode(out, skip_special_tokens=True)[0]

def answer_from_image_and_query(
    image_path, user_query, json_path, index_path, meta_path, pdf_path_for_source=None
):
    # 1) smart query
    image_name = os.path.basename(image_path)
    smart_query = build_retrieval_query_from_json(image_name, user_query, json_path)

    # 2) retrieval
    index, documents, metadata = load_index(index_path, meta_path)
    embedder = load_embedder()
    hits = retrieve_context(index, embedder, documents, metadata, smart_query, top_k=TOP_K)
    context = "\n\n".join([h["text"] for h in hits]) if hits else ""

    # 3) VLM
    img = Image.open(image_path).convert("RGB")
    processor, model = load_qwen()
    prompt = make_prompt(context, user_query)
    resp = generate_answer(img, prompt, processor, model)

    # Optional: show what was used
    preview = "\n".join([f"[{h['rank']}] {h['source']} (d={h['distance']:.3f})" for h in hits])
    return resp, preview, smart_query
