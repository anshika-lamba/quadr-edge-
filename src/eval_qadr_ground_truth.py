import csv
import json
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
from qadr import apply_qadr
from perturb import apply_poisson_gaussian_noise

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_raw_rgb(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    return np.array(img).astype(np.float32) / 255.0

def to_model_tensor(rgb_array):
    norm = (rgb_array - MEAN) / STD
    return np.transpose(norm, (2, 0, 1))[np.newaxis, :].astype(np.float32)

def eval_ground_truth():
    print("=" * 70)
    print("QADR Evaluation: Real ImageNet Ground-Truth Validation (N=950)")
    print("=" * 70)
    
    # Calibrated parameter vector: alpha, beta, gamma_min, gamma_max
    alpha, beta, g_min, g_max = 1.0376, 0.0002, 0.0002, 0.9034
    
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session_fp32 = ort.InferenceSession("models/mobilenetv4_small_fp32.onnx", opts)
    session_int8 = ort.InferenceSession("models/mobilenetv4_small_int8.onnx", opts)
    
    in_fp32, out_fp32 = session_fp32.get_inputs()[0].name, session_fp32.get_outputs()[0].name
    in_int8, out_int8 = session_int8.get_inputs()[0].name, session_int8.get_outputs()[0].name
    
    severities = [0.00, 0.01, 0.03, 0.05, 0.10]
    
    with open("data/manifest.csv", "r") as f:
        val_records = list(csv.DictReader(f))[50:]  # N=950 validation set
        
    results = {sig: {'fp32': 0, 'int8': 0, 'qadr': 0, 'total': 0} for sig in severities}
    
    for idx, row in enumerate(val_records):
        raw_rgb = load_raw_rgb(row['file_path'])
        gt_label = int(row['ground_truth_label'])
        
        for sig in severities:
            noisy_rgb = apply_poisson_gaussian_noise(raw_rgb, sigma=sig) if sig > 0 else raw_rgb
            
            # FP32 Pass
            fp_tensor = to_model_tensor(noisy_rgb)
            res_fp = session_fp32.run([out_fp32], {in_fp32: fp_tensor})[0]
            pred_fp = int(np.argmax(res_fp))
            
            # INT8 Baseline Pass
            int_tensor = to_model_tensor(noisy_rgb)
            res_int = session_int8.run([out_int8], {in_int8: int_tensor})[0]
            pred_int = int(np.argmax(res_int))
            
            # QADR + INT8 Pass
            qadr_rgb = apply_qadr(noisy_rgb, alpha, beta, g_min, g_max)
            qadr_tensor = to_model_tensor(qadr_rgb)
            res_qadr = session_int8.run([out_int8], {in_int8: qadr_tensor})[0]
            pred_qadr = int(np.argmax(res_qadr))
            
            results[sig]['total'] += 1
            if pred_fp == gt_label: results[sig]['fp32'] += 1
            if pred_int == gt_label: results[sig]['int8'] += 1
            if pred_qadr == gt_label: results[sig]['qadr'] += 1
            
        if (idx + 1) % 100 == 0 or (idx + 1) == len(val_records):
            print(f"[+] Evaluated {idx + 1}/{len(val_records)} validation scenes...")

    print("\n" + "=" * 75)
    print("      REAL GROUND-TRUTH VALIDATION SUMMARY (ImageNet Subset N=950)")
    print("=" * 75)
    print(f"{'Severity (σ)':<12} | {'FP32 Top-1':<10} | {'INT8 Baseline':<14} | {'QADR + INT8':<14} | {'Gain (pp)':<10}")
    print("-" * 75)
    
    md_lines = ["# Ground-Truth Validated Metrics Summary (N=950)\n",
                "| Noise Severity (σ) | FP32 Top-1 Acc | INT8 Baseline | QADR + INT8 | Net Gain |",
                "| :---: | :---: | :---: | :---: | :---: |"]
    
    for sig in severities:
        tot = results[sig]['total']
        fp_acc = (results[sig]['fp32'] / tot) * 100.0
        int_acc = (results[sig]['int8'] / tot) * 100.0
        qadr_acc = (results[sig]['qadr'] / tot) * 100.0
        gain = qadr_acc - int_acc
        
        print(f"{sig:<12.2f} | {fp_acc:<9.1f}% | {int_acc:<13.1f}% | {qadr_acc:<13.1f}% | {gain:<+9.1f}pp")
        md_lines.append(f"| {sig:.2f} | {fp_acc:.1f}% | {int_acc:.1f}% | **{qadr_acc:.1f}%** | **{gain:+.1f}pp** |")
        
    print("=" * 75 + "\n")
    
    with open("results/metrics_summary_real.md", "w") as f:
        f.write("\n".join(md_lines))
        
    print("[+] Verified metrics saved to results/metrics_summary_real.md")

if __name__ == "__main__":
    eval_ground_truth()
