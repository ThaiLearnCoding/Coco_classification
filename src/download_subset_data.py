import json
import os
import requests
from PIL import Image
from zipfile import ZipFile
from tqdm import tqdm
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

# --- CẤU HÌNH CHUNG ---
ANNOTATION_PATH = 'instances_train2017.json'
ZIP_NAME = 'coco_multimodal_subset.zip'
TARGET_SIZE = (336, 336)
TARGET_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
    'bus', 'train', 'truck', 'boat', 'bird', 
    'cat', 'dog', 'horse', 'sheep', 'cow'
]

# --- 1. CHUẨN BỊ DỮ LIỆU CHUNG (Chạy một lần) ---
print("Loading annotations...")
with open(ANNOTATION_PATH, 'r') as f:
    data = json.load(f)

cat_name_to_id = {cat['name']: cat['id'] for cat in data['categories'] if cat['name'] in TARGET_CLASSES}
cat_id_to_name = {v: k for k, v in cat_name_to_id.items()}
target_ids = set(cat_name_to_id.values())

img_id_to_meta = {img['id']: {'url': img['coco_url'], 'file_name': img['file_name']} for img in data['images']}
img_to_labels = {}

for ann in data['annotations']:
    if ann['category_id'] in target_ids:
        img_id = ann['image_id']
        if img_id not in img_to_labels:
            img_to_labels[img_id] = set()
        img_to_labels[img_id].add(cat_id_to_name[ann['category_id']])

final_img_ids = list(img_to_labels.keys())[:12000] # Biến này giờ đã nằm ở Global Scope
metadata_subset = []

# --- 2. HÀM XỬ LÝ TỪNG ẢNH (Dùng cho ThreadPool) ---
def download_and_resize(img_id):
    meta = img_id_to_meta[img_id]
    labels = list(img_to_labels[img_id])
    try:
        response = requests.get(meta['url'], timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        
        # Trả về dữ liệu để luồng chính ghi vào Zip (tránh xung đột ghi file)
        img_io = BytesIO()
        img.save(img_io, format='JPEG', quality=85) # Giảm nhẹ quality để nén nhanh hơn
        return meta['file_name'], img_io.getvalue(), labels
    except:
        return None

# --- 3. THỰC THI ĐA LUỒNG ---
print(f"Starting Multi-threaded download (20 workers) for {len(final_img_ids)} images...")

with ZipFile(ZIP_NAME, 'w') as zip_file:
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Dùng list(tqdm(...)) để theo dõi tiến trình
        results = list(tqdm(executor.map(download_and_resize, final_img_ids), total=len(final_img_ids)))

    # --- 4. TỔNG HỢP KẾT QUẢ VÀO FILE ZIP ---
    print("Saving to Zip and creating metadata...")
    for res in results:
        if res is not None:
            file_name, img_bytes, labels = res
            zip_file.writestr(f"images/{file_name}", img_bytes)
            metadata_subset.append({
                'file_name': file_name,
                'labels': labels
            })

    # Lưu file nhãn vào trong zip
    with open('metadata.json', 'w') as f:
        json.dump(metadata_subset, f)
    zip_file.write('metadata.json', 'metadata.json')

print(f"\nDone! Saved {len(metadata_subset)} images to {ZIP_NAME}")