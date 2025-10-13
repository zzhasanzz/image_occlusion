import easyocr, os, cv2, re, json, numpy as np
from collections import defaultdict

def is_number_bubble(t):
    t = t.strip()
    if re.fullmatch(r"[0-9]{1,3}", t): return True
    if re.fullmatch(r"[ivxlcdmIVXLCDM]{1,4}", t): return True
    if any('\u2460' <= ch <= '\u2473' for ch in t): return True
    if len(t) == 1 and t.isalpha(): return True
    return False

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

def process_image(image_path, reader):
    img = cv2.imread(image_path)
    if img is None:
        return []

    results = reader.readtext(img)
    boxes = []
    for (poly, text, conf) in results:
        if not text.strip(): continue
        p = np.array(poly, dtype=int)
        x1, y1 = p[:,0].min(), p[:,1].min()
        x2, y2 = p[:,0].max(), p[:,1].max()
        boxes.append({"bbox": (x1, y1, x2, y2), "text": text.strip(), "conf": float(conf)})

    if not boxes: return []

    for b in boxes:
        x1,y1,x2,y2 = b["bbox"]
        b["w"], b["h"] = x2-x1, y2-y1
        b["cx"], b["cy"] = (x1+x2)/2, (y1+y2)/2

    # parameters
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

    merged=[]
    for ids in groups:
        g=[boxes[k] for k in ids]
        xs=[b["bbox"][0] for b in g]; ys=[b["bbox"][1] for b in g]
        xe=[b["bbox"][2] for b in g]; ye=[b["bbox"][3] for b in g]
        merged.append({"bbox":(int(min(xs)),int(min(ys)),int(max(xe)),int(max(ye))),
                       "text":" ".join([b["text"] for b in sorted(g,key=lambda b:(b['cy'],b['cx']))])})
    return merged

def detect_labels(folder_path, output_json="merged_boxes_all.json"):
    reader = easyocr.Reader(['en'], gpu=True)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    all_data = {}
    for fname in sorted(image_files):
        path=os.path.join(folder_path,fname)
        merged=process_image(path,reader)
        all_data[fname]=merged
        print(f"✅ Processed {fname} ({len(merged)} regions)")
    with open(output_json,"w") as f:
        json.dump(all_data,f,indent=2)
    print(f"📄 Saved to {output_json}")
