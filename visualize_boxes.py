import cv2, os, json, matplotlib.pyplot as plt

def visualize_boxes(images_folder, json_path, output_folder="annotated"):
    os.makedirs(output_folder, exist_ok=True)
    with open(json_path,"r") as f:
        data=json.load(f)

    for fname, boxes in data.items():
        img_path=os.path.join(images_folder,fname)
        img=cv2.imread(img_path)
        if img is None: continue
        for b in boxes:
            x1,y1,x2,y2=b["bbox"]
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(img,b["text"][:25],(x1,max(y1-8,10)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        out_path=os.path.join(output_folder,fname)
        cv2.imwrite(out_path,img)
        print(f"💾 Saved annotated: {out_path}")
