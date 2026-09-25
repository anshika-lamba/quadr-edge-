import csv
import math
import numpy as np

def compute_wilson_ci(k, n, confidence=0.95):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    z = 1.96  # 95% Confidence
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - margin) * 100.0, min(100.0, centre + margin) * 100.0)

def analyze_results(csv_path="results/audit_raw.csv", output_md="results/audit_summary.md"):
    data = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prec = row['precision']
            sig = float(row['noise_severity'])
            correct = int(row['is_correct'])
            
            key = (prec, sig)
            if key not in data:
                data[key] = {'correct': 0, 'total': 0}
            data[key]['correct'] += correct
            data[key]['total'] += 1

    severities = sorted(list(set(k[1] for k in data.keys())))
    
    lines = []
    lines.append("# QADR Phase 3: Native On-Device Robustness Audit Summary\n")
    lines.append("| Noise Severity (σ) | FP32 Top-1 Acc (95% CI) | INT8 Top-1 Acc (95% CI) | Robustness Delta (Δ) |")
    lines.append("| :---: | :---: | :---: | :---: |")
    
    print("\n" + "=" * 75)
    print("      QADR Phase 3: Empirical Robustness Audit Results")
    print("=" * 75)
    print(f"{'Severity (σ)':<12} | {'FP32 Accuracy (95% CI)':<24} | {'INT8 Accuracy (95% CI)':<24} | {'Delta (Δ)':<10}")
    print("-" * 75)
    
    for sig in severities:
        fp_stats = data.get(('FP32', sig), {'correct': 0, 'total': 1})
        int_stats = data.get(('INT8', sig), {'correct': 0, 'total': 1})
        
        fp_acc = (fp_stats['correct'] / fp_stats['total']) * 100.0
        int_acc = (int_stats['correct'] / int_stats['total']) * 100.0
        delta = fp_acc - int_acc
        
        fp_ci = compute_wilson_ci(fp_stats['correct'], fp_stats['total'])
        int_ci = compute_wilson_ci(int_stats['correct'], int_stats['total'])
        
        fp_str = f"{fp_acc:.1f}% [{fp_ci[0]:.1f}%, {fp_ci[1]:.1f}%]"
        int_str = f"{int_acc:.1f}% [{int_ci[0]:.1f}%, {int_ci[1]:.1f}%]"
        
        print(f"{sig:<12.2f} | {fp_str:<24} | {int_str:<24} | {delta:<+10.1f}%")
        lines.append(f"| {sig:.2f} | {fp_str} | {int_str} | **{delta:+.1f}%** |")
        
    print("=" * 75 + "\n")
    
    with open(output_md, 'w') as f:
        f.write('\n'.join(lines))
        
    print(f"[+] Markdown summary saved to: {output_md}")

if __name__ == "__main__":
    analyze_results()
