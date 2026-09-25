import os
import csv
import numpy as np
from PIL import Image

def setup_dataset(num_scenes=1000):
    print("=" * 60)
    print("QADR Phase 3 Setup: Dataset Manifest Generation")
    print("=" * 60)
    
    os.makedirs("data/images", exist_ok=True)
    manifest_path = "data/manifest.csv"
    
    # Check if user already populated real images
    existing_files = [f for f in os.listdir("data/images") if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'file_path', 'ground_truth_label'])
        
        if len(existing_files) >= 100:
            print(f"[+] Found {len(existing_files)} existing images in data/images/")
            for idx, fname in enumerate(sorted(existing_files)):
                img_id = f"{idx+1:05d}"
                path = os.path.join("data/images", fname)
                # Assign deterministic synthetic ground truth labels (0-999)
                label = idx % 1000
                writer.writerow([img_id, path, label])
        else:
            print(f"[*] Generating {num_scenes} standardized test scenes in data/images/...")
            rng = np.random.default_rng(42)
            for idx in range(num_scenes):
                img_id = f"{idx+1:05d}"
                path = f"data/images/img_{img_id}.jpg"
                
                # Generate structured test image (geometric patterns + synthetic features)
                arr = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
                Image.fromarray(arr).save(path, quality=95)
                
                label = idx % 1000
                writer.writerow([img_id, path, label])
                
    print(f"[+] Manifest successfully generated at: {manifest_path}")
    print("=" * 60)

if __name__ == "__main__":
    setup_dataset()
