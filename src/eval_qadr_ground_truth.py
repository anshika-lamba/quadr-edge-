import csv
import json
import math
import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from qadr import apply_qadr
from perturb import apply_poisson_gaussian_noise

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def compute_wilson_ci(k, n, confidence=0.95):
    if n == 0: return (0.0, 0.0)
    p = k / n
    z = 1.96
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - margin) * 100.0, min(100.0, centre + margin) * 100.0)

def compute_mcnemar_p_value(b, c):
    if (b + c) == 0: return 1.0
    chi2 = (abs(b - c) - 1.0)**2 / (b + c)
    p_val = math.erfc(math.sqrt(chi2 / 2.0))
    return p_val

def load_raw_rgb(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    return np.array(img).astype(np.float32) / 255.0

def to_model_tensor(rgb_array):
    norm = (rgb_array - MEAN) / STD
    return np.transpose(norm, (2, 0, 1))[np.newaxis, :].astype(np.float32)

def eval_ground_truth():
    print("=" * 75)
    print("QADR Evaluation: Ground-Truth Image Validation (Out-of-Sample N=90)")
    print("=" * 75)
    
    param_path = "results/calibrated_params.json"
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"Calibrated parameters file not found at {param_path}!")
        
    with open(param_path, "r") as f:
        params = json.load(f)
        
    alpha, beta, g_min, g_max = params['alpha'], params['beta'], params['gamma_min'], params['gamma_max']
    print(f"[*] Loaded Dynamic Calibration Vector from {param_path}:")
    print(f"    alpha={alpha:.6f}, beta={beta:.6f}, gamma_min={g_min:.6f}, gamma_max={g_max:.6f}")
    print("-" * 75)
    
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session_fp32 = ort.InferenceSession("models/mobilenetv4_small_fp32.onnx", opts)
    session_int8 = ort.InferenceSession("models/mobilenetv4_small_int8.onnx", opts)
    
    in_fp32, out_fp32 = session_fp32.get_inputs()[0].name, session_fp32.get_outputs()[0].name
    in_int8, out_int8 = session_int8.get_inputs()[0].name, session_int8.get_outputs()[0].name
    
    severities = [0.00, 0.01, 0.03, 0.05, 0.10]
    
    with open("data/manifest.csv", "r") as f:
        all_records = list(csv.DictReader(f))
        val_records = all_records[10:]  # Images 11..100 (N=90 validation split)
        
    print(f"[*] Loaded N={len(val_records)} validation scenes.")
        
    results = {sig: {'fp32': 0, 'int8': 0, 'qadr': 0, 'total': 0, 'b': 0, 'c': 0} for sig in severities}
    
    for idx, row in enumerate(val_records):
        raw_rgb = load_raw_rgb(row['file_path'])
        gt_label = int(row['ground_truth_label'])
        
        for sig in severities:
            noisy_rgb = apply_poisson_gaussian_noise(raw_rgb, sigma=sig) if sig > 0 else raw_rgb
            
            # FP32
            fp_tensor = to_model_tensor(noisy_rgb)
            res_fp = session_fp32.run([out_fp32], {in_fp32: fp_tensor})[0]
            pred_fp = int(np.argmax(res_fp))
            
            # INT8 Baseline
            int_tensor = to_model_tensor(noisy_rgb)
            res_int = session_int8.run([out_int8], {in_int8: int_tensor})[0]
            pred_int = int(np.argmax(res_int))
            
            # QADR + INT8
            qadr_rgb = apply_qadr(noisy_rgb, alpha, beta, g_min, g_max)
            qadr_tensor = to_model_tensor(qadr_rgb)
            res_qadr = session_int8.run([out_int8], {in_int8: qadr_tensor})[0]
            pred_qadr = int(np.argmax(res_qadr))
            
            correct_int = (pred_int == gt_label)
            correct_qadr = (pred_qadr == gt_label)
            
            results[sig]['total'] += 1
            if pred_fp == gt_label: results[sig]['fp32'] += 1
            if correct_int: results[sig]['int8'] += 1
            if correct_qadr: results[sig]['qadr'] += 1
            
            if correct_int and not correct_qadr:
                results[sig]['b'] += 1
            elif not correct_int and correct_qadr:
                results[sig]['c'] += 1

        if (idx + 1) % 15 == 0 or (idx + 1) == len(val_records):
            print(f"  [+] Progress: Evaluated {idx + 1}/{len(val_records)} validation scenes...")

    print("\n" + "=" * 80)
    print("      REAL GROUND-TRUTH VALIDATION SUMMARY (Out-of-Sample N=90)")
    print("=" * 80)
    print(f"{'Severity (σ)':<12} | {'FP32 Top-1 (95% CI)':<22} | {'INT8 Baseline':<14} | {'QADR + INT8':<14} | {'Gain':<8} | {'p-value':<8}")
    print("-" * 80)
    
    md_lines = ["# Ground-Truth Validated Metrics Summary\n",
                "| Noise Severity (σ) | FP32 Top-1 Acc (95% CI) | INT8 Baseline | QADR + INT8 | Net Gain | McNemar p-value |",
                "| :---: | :---: | :---: | :---: | :---: | :---: |"]
    
    for sig in severities:
        tot = results[sig]['total']
        if tot == 0: continue
        
        fp_acc = (results[sig]['fp32'] / tot) * 100.0
        int_acc = (results[sig]['int8'] / tot) * 100.0
        qadr_acc = (results[sig]['qadr'] / tot) * 100.0
        gain = qadr_acc - int_acc
        
        fp_ci = compute_wilson_ci(results[sig]['fp32'], tot)
        p_val = compute_mcnemar_p_value(results[sig]['b'], results[sig]['c'])
        
        fp_str = f"{fp_acc:.1f}% [{fp_ci[0]:.1f}%, {fp_ci[1]:.1f}%]"
        gain_str = f"{gain:+.1f}pp"
        p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
        
        print(f"{sig:<12.2f} | {fp_str:<22} | {int_acc:<13.1f}% | {qadr_acc:<13.1f}% | {gain_str:<8} | {p_str:<8}")
        md_lines.append(f"| {sig:.2f} | {fp_str} | {int_acc:.1f}% | **{qadr_acc:.1f}%** | **{gain_str}** | p={p_str} |")
        
    print("=" * 80 + "\n")
    
    with open("results/metrics_summary_real.md", "w") as f:
        f.write("\n".join(md_lines))
        
    print("[+] Verified metrics and statistical tests saved to results/metrics_summary_real.md")

if __name__ == "__main__":
    eval_ground_truth()
