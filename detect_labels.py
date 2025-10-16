import easyocr, os, cv2, re, json, numpy as np
import torch
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# Text filters
# ─────────────────────────────────────────────────────────────

FIGURE_RE = re.compile(
    r"""^\s*
        (fig(?:ure)?\.?)      # Fig / Figure / Fig.
        \s*[:\-]?\s*
        [0-9]+(?:\.[0-9]+)?   # 2 or 2.3
        [a-zA-Z]?             # optional A/B
        (?:\s*[:\-–]\s*.*)?   # optional caption text
        $""",
    re.IGNORECASE | re.VERBOSE,
)

PANEL_RE = re.compile(r"^\s*\(?[a-dA-D]\)?\s*$")  # a, (a), B, (C)

MEASUREMENT_RE = re.compile(
    r"""^\s*
        [\+\-]?\d+(?:\.\d+)?          # number
        (?:\s*[~–\-]\s*[\+\-]?\d+(?:\.\d+)?)?  # optional range like -55 ~70 / 70–90
        (?:\s*(?:mm|cm|m|°|deg|%))?   # optional unit
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
)

ROMAN_RE = re.compile(r"^[ivxlcdmIVXLCDM]{1,4}$")
CIRCLED_RANGE = ("\u2460", "\u2473")  # ①–⑳

def is_circled_digit(text: str) -> bool:
    return any("\u2460" <= ch <= "\u2473" for ch in text)

def clean_text(t: str) -> str:
    # normalize common OCR artifacts
    return (
        t.replace("\u2013", "-")  # en dash
         .replace("\u2014", "-")  # em dash
         .replace("\ufb01", "fi")
         .replace("\ufb02", "fl")
         .strip()
    )

def is_number_bubble(t: str) -> bool:
    t = t.strip()
    if re.fullmatch(r"[0-9]{1,3}", t): return True
    if ROMAN_RE.fullmatch(t): return True
    if is_circled_digit(t): return True
    if PANEL_RE.fullmatch(t): return True  # (a), a
    if len(t) == 1 and t.isalpha(): return True
    return False

def is_invalid_label(t: str) -> bool:
    """
    Returns True when the text should be skipped from labeling/flashcards.
    """
    t = clean_text(t)
    # 1) very short (after removing spaces/punct)
    letters = re.sub(r"[^A-Za-z0-9]", "", t)
    if len(letters) < 3:
        return True
    # 2) figure captions
    if FIGURE_RE.match(t):
        return True
    # 3) plain numeric/measurement/range-like
    if MEASUREMENT_RE.match(t):
        return True
    # 4) panel letters / bubbles
    if is_number_bubble(t):
        return True
    return False

# ─────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────

def horiz_gap(a,b):
    ax1,_,ax2,_=a["bbox"]; bx1,_,bx2,_=b["bbox"]
    if ax2 <= bx1: return bx1-ax2
    if bx2 <= ax1: return ax1-bx2
    return 0

def vert_gap(a,b):
    ay1,ay2=a["bbox"][1],a["bbox"][3]; by1,by2=b["bbox"][1],b["bbox"][3]
    if ay2 <= by1: return by1-ay2
    if by2 <= ay1: return ay1-by2
    return 0

def horiz_overlap_ratio(a,b):
    ax1,_,ax2,_=a["bbox"]; bx1,_,bx2,_=b["bbox"]
    inter=max(0,min(ax2,bx2)-max(ax1,bx1))
    denom=max(1,min(ax2-ax1,bx2-bx1))
    return inter/denom

# ─────────────────────────────────────────────────────────────
# Core
# ─────────────────────────────────────────────────────────────

def process_image(image_path, reader, conf_min=0.4, min_h_px=10, min_area_ratio=0.00002):
    """
    Returns merged label boxes after filtering:
      - OCR confidence >= conf_min
      - box height >= min_h_px
      - box area >= min_area_ratio * image_area
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    H_img, W_img = img.shape[:2]
    min_area = max(1, int(min_area_ratio * W_img * H_img))

    results = reader.readtext(img)
    boxes = []
    for (poly, text, conf) in results:
        text = clean_text(text)
        if not text:
            continue
        if conf < conf_min:
            continue

        p = np.array(poly, dtype=int)
        x1, y1 = max(0, p[:,0].min()), max(0, p[:,1].min())
        x2, y2 = min(W_img, p[:,0].max()), min(H_img, p[:,1].max())
        w, h = x2 - x1, y2 - y1
        if h < min_h_px or (w * h) < min_area:
            continue

        boxes.append({"bbox": (x1, y1, x2, y2), "text": text, "conf": float(conf)})

    if not boxes:
        return []

    for b in boxes:
        x1,y1,x2,y2 = b["bbox"]
        b["w"], b["h"] = x2-x1, y2-y1
        b["cx"], b["cy"] = (x1+x2)/2, (y1+y2)/2

    # graph build
    SAME_LINE_Y_TOL, STACK_MIN_OVERLAP, PROX_EPS_L1, BUBBLE_RADIUS, VERT_ISO_FACTOR = 0.6, 0.35, 3.8, 3.0, 0.25
    med_h = max(8.0, float(np.median([b["h"] for b in boxes])))
    N=len(boxes); edges=[[] for _ in range(N)]

    for i in range(N):
        bi=boxes[i]
        for j in range(i+1,N):
            bj=boxes[j]
            H=max(bi["h"],bj["h"])
            base_gap=0.5*H
            length_factor=min(1.0+max(bi["w"],bj["w"])/(8*H),3.5)
            adaptive_hgap=base_gap*length_factor
            y_penalty=1.0+abs(bi["cy"]-bj["cy"])/(1.2*H)
            adaptive_hgap/=y_penalty

            same_line=abs(bi["cy"]-bj["cy"])<=SAME_LINE_Y_TOL*H and horiz_gap(bi,bj)<=adaptive_hgap
            stacked=horiz_overlap_ratio(bi,bj)>=STACK_MIN_OVERLAP and vert_gap(bi,bj)<=1.4*H
            l1_norm=(abs(bi["cx"]-bj["cx"])+abs(bi["cy"]-bj["cy"]))/med_h
            proximity=l1_norm<=PROX_EPS_L1

            bubble=False
            if is_number_bubble(bi["text"]) or is_number_bubble(bj["text"]):
                dx=(bi["cx"]-bj["cx"])/med_h; dy=(bi["cy"]-bj["cy"])/med_h
                bubble=(dx*dx+dy*dy)**0.5<=BUBBLE_RADIUS

            vgap=vert_gap(bi,bj); isolate=vgap>VERT_ISO_FACTOR*H
            if (same_line or stacked or proximity or bubble) and not isolate:
                edges[i].append(j); edges[j].append(i)

    # components
    visited=[False]*N; groups=[]
    for i in range(N):
        if visited[i]: continue
        comp=[]; stack=[i]; visited[i]=True
        while stack:
            k=stack.pop()
            comp.append(k)
            for nb in edges[k]:
                if not visited[nb]:
                    visited[nb]=True; stack.append(nb)
        groups.append(comp)

    # merge → filter by text rules
    merged=[]
    for ids in groups:
        g=[boxes[k] for k in ids]
        xs=[b["bbox"][0] for b in g]; ys=[b["bbox"][1] for b in g]
        xe=[b["bbox"][2] for b in g]; ye=[b["bbox"][3] for b in g]
        text=" ".join([b["text"] for b in sorted(g,key=lambda b:(b['cy'],b['cx']))]).strip()

        # FINAL FILTERS (apply on merged text)
        if is_invalid_label(text):
            continue

        merged.append({
            "bbox":(int(min(xs)),int(min(ys)),int(max(xe)),int(max(ye))),
            "text":text
        })
    return merged

def detect_labels(folder_path, output_json="merged_boxes_all.json"):
    use_gpu = torch.cuda.is_available()
    print(f"🔍 torch.cuda.is_available() => {use_gpu}")
    reader = easyocr.Reader(['en'], gpu=use_gpu)

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    all_data = {}
    for fname in sorted(image_files):
        path=os.path.join(folder_path,fname)
        merged=process_image(path,reader)
        all_data[fname]=merged
        print(f"✅ Processed {fname} ({len(merged)} valid regions)")
    with open(output_json,"w") as f:
        json.dump(all_data,f,indent=2)
    print(f"📄 Saved to {output_json}")
