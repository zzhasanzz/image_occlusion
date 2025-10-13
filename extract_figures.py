import fitz, torch, os
from PIL import Image
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection

def setup_model(device="cpu"):
    processor = AutoImageProcessor.from_pretrained("Aryn/deformable-detr-DocLayNet")
    model = DeformableDetrForObjectDetection.from_pretrained("Aryn/deformable-detr-DocLayNet").to(device)
    return processor, model

def extract_pictures(page_img, page_idx, processor, model, output_dir, device):
    inputs = processor(images=page_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([page_img.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

    count = 0
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        name = model.config.id2label[label.item()]
        if name.lower() == "picture" and score > 0.7:
            x1, y1, x2, y2 = map(int, box.tolist())
            crop = page_img.crop((x1, y1, x2, y2))
            out_path = f"{output_dir}/page{page_idx+1:03d}_fig{count+1:02d}.png"
            crop.save(out_path)
            count += 1
    return count

def extract_figures_from_pdf(pdf_path, output_dir, page_range=None, zoom=2.5):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = setup_model(device)
    doc = fitz.open(pdf_path)

    total = 0
    if page_range is None:
        page_range = range(len(doc))

    for i in page_range:
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        n = extract_pictures(img, i, processor, model, output_dir, device)
        print(f"📄 Page {i+1}: {n} figure(s) saved.")
        total += n

    print(f"\n🎯 Done! {total} total figures saved in '{output_dir}/'")
