import os
import csv
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
from perturb import perturb_image

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    # ImageNet Mean & Std Normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    # Transpose to Channel-First (1, 3, 224, 224)
    tensor = np.transpose(arr, (2, 0, 1))[np.newaxis, :].astype(np.float32)
    return tensor

def execute_audit(manifest_path, fp32_path, int8_path, output_csv):
    print("=" * 65)
    print("QADR Phase 3: On-Device Audit (Reference Oracle Alignment)")
    print("=" * 65)
    
    os.makedirs("results", exist_ok=True)
    
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session_fp32 = ort.InferenceSession(fp32_path, opts)
    session_int8 = ort.InferenceSession(int8_path, opts)
    
    in_fp32, out_fp32 = session_fp32.get_inputs()[0].name, session_fp32.get_outputs()[0].name
    in_int8, out_int8 = session_int8.get_inputs()[0].name, session_int8.get_outputs()[0].name
    
    severities = [0.00, 0.01, 0.03, 0.05, 0.10]
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'image_id', 'precision', 'noise_severity', 'pred_class', 'ground_truth', 'is_correct', 'latency_ms'])
        
        with open(manifest_path, 'r') as mf:
            records = list(csv.DictReader(mf))
            total = len(records)
            
            for idx, row in enumerate(records):
                image_id, path = row['image_id'], row['file_path']
                base_tensor = preprocess_image(path)
                
                # Oracle Pass: Use clean FP32 prediction as the reference ground truth
                oracle_res = session_fp32.run([out_fp32], {in_fp32: base_tensor})[0]
                oracle_label = int(np.argmax(oracle_res))
                
                for sig in severities:
                    noisy_tensor = perturb_image(base_tensor, "poisson_gaussian", severity=sig) if sig > 0 else base_tensor
                    
                    # FP32 Pass
                    t0 = time.perf_counter()
                    res_fp32 = session_fp32.run([out_fp32], {in_fp32: noisy_tensor})[0]
                    lat_fp32 = (time.perf_counter() - t0) * 1000.0
                    pred_fp32 = int(np.argmax(res_fp32))
                    writer.writerow([time.time(), image_id, 'FP32', sig, pred_fp32, oracle_label, int(pred_fp32 == oracle_label), f"{lat_fp32:.3f}"])
                    
                    # INT8 Pass
                    t0 = time.perf_counter()
                    res_int8 = session_int8.run([out_int8], {in_int8: noisy_tensor})[0]
                    lat_int8 = (time.perf_counter() - t0) * 1000.0
                    pred_int8 = int(np.argmax(res_int8))
                    writer.writerow([time.time(), image_id, 'INT8', sig, pred_int8, oracle_label, int(pred_int8 == oracle_label), f"{lat_int8:.3f}"])
                
                if (idx + 1) % 100 == 0 or (idx + 1) == total:
                    print(f"[+] Evaluated {idx + 1}/{total} scenes...")

    print("=" * 65)
    print(f"[STATUS] Audit complete. Logged to {output_csv}")
    print("=" * 65)

if __name__ == "__main__":
    execute_audit("data/manifest.csv", "models/mobilenetv4_small_fp32.onnx", "models/mobilenetv4_small_int8.onnx", "results/audit_raw.csv")
