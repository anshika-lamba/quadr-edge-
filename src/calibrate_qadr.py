import os
import csv
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.optimize import minimize
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

def calibrate():
    print("=" * 65)
    print("QADR Phase 4: On-Device Calibration (Fixed Order)")
    print("=" * 65)
    
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session_fp32 = ort.InferenceSession("models/mobilenetv4_small_fp32.onnx", opts)
    session_int8 = ort.InferenceSession("models/mobilenetv4_small_int8.onnx", opts)
    
    in_fp32, out_fp32 = session_fp32.get_inputs()[0].name, session_fp32.get_outputs()[0].name
    in_int8, out_int8 = session_int8.get_inputs()[0].name, session_int8.get_outputs()[0].name
    
    print("[*] Pre-computing calibration samples (N=15)...")
    calib_data = []
    with open("data/manifest.csv", 'r') as f:
        records = list(csv.DictReader(f))[:15]
        for row in records:
            raw_rgb = load_raw_rgb(row['file_path'])
            
            # Oracle prediction on clean image
            clean_tensor = to_model_tensor(raw_rgb)
            oracle_res = session_fp32.run([out_fp32], {in_fp32: clean_tensor})[0]
            oracle_label = int(np.argmax(oracle_res))
            
            # Apply fixed noise in [0, 1] RGB space
            np.random.seed(42)
            noisy_rgb = apply_poisson_gaussian_noise(raw_rgb, sigma=0.05)
            calib_data.append((noisy_rgb, oracle_label))
            
    step_counter = [0]
    def objective_loss(theta):
        alpha, beta, g_min, g_max = theta
        total_loss = 0.0
        
        for noisy_rgb, oracle_label in calib_data:
            qadr_rgb = apply_qadr(noisy_rgb, alpha, beta, g_min, g_max)
            input_tensor = to_model_tensor(qadr_rgb)
            
            logits = session_int8.run([out_int8], {in_int8: input_tensor})[0][0]
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / np.sum(exp_l)
            total_loss += -np.log(probs[oracle_label] + 1e-7)
            
        step_counter[0] += 1
        avg_loss = total_loss / len(calib_data)
        if step_counter[0] % 5 == 0:
            print(f"  -> Step {step_counter[0]:03d} | Loss: {avg_loss:.4f} | alpha: {alpha:.3f}, beta: {beta:.3f}")
        return avg_loss

    init_theta = [1.0, 0.0, 0.0, 1.0]
    res = minimize(objective_loss, init_theta, method='Nelder-Mead', options={'maxiter': 40})
    
    opt_params = {
        "alpha": float(res.x[0]),
        "beta": float(res.x[1]),
        "gamma_min": float(res.x[2]),
        "gamma_max": float(res.x[3]),
        "calibration_loss": float(res.fun)
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/calibrated_params.json", "w") as f:
        json.dump(opt_params, f, indent=2)
        
    print("-" * 65)
    print(f"[+] Calibration Complete in {step_counter[0]} steps!")
    print(f"    alpha     : {opt_params['alpha']:.4f}")
    print(f"    beta      : {opt_params['beta']:.4f}")
    print(f"    gamma_min : {opt_params['gamma_min']:.4f}")
    print(f"    gamma_max : {opt_params['gamma_max']:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    calibrate()
