import cv2, os, json, matplotlib.pyplot as plt

def apply_flashcard_mask(img, boxes, highlight_idx):
    overlay = img.copy()
    dark_orange = (0,110,220)
    highlight_fill = (0,180,255)
    crimson = (30,20,180)
    alpha=1.0

    for b in boxes:
        x1,y1,x2,y2=b["bbox"]
        cv2.rectangle(overlay,(x1,y1),(x2,y2),dark_orange,-1)

    (x1,y1,x2,y2)=boxes[highlight_idx]["bbox"]
    cv2.rectangle(overlay,(x1,y1),(x2,y2),highlight_fill,-1)
    blended=cv2.addWeighted(overlay,alpha,img,1-alpha,0)
    cv2.rectangle(blended,(x1,y1),(x2,y2),crimson,2)
    return blended

def generate_flashcards(images_folder, json_path, output_folder="flashcards"):
    os.makedirs(output_folder, exist_ok=True)
    with open(json_path,"r") as f:
        data=json.load(f)

    for img_idx, (fname, boxes) in enumerate(list(data.items())):
        img_path=os.path.join(images_folder,fname)
        img=cv2.imread(img_path)
        if img is None: continue
        for i in range(len(boxes)):
            masked_img=apply_flashcard_mask(img,boxes,i)
            out_name=f"{os.path.splitext(fname)[0]}_q{i+1}.png"
            out_path=os.path.join(output_folder,out_name)
            cv2.imwrite(out_path,masked_img)
            print(f"✅ Flashcard saved: {out_name}")

    print(f"\n🎯 Done! Flashcards saved in '{output_folder}/'")
