import numpy as np
import onnxruntime as ort
from PIL import Image

def diagnose():
    print("=" * 65)
    print("QADR Diagnostics: Identifying INT8 Model Input Format")
    print("=" * 65)
    
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session_fp32 = ort.InferenceSession("models/mobilenetv4_small_fp32.onnx", opts)
    session_int8 = ort.InferenceSession("models/mobilenetv4_small_int8.onnx", opts)
    
    in_fp32, out_fp32 = session_fp32.get_inputs()[0].name, session_fp32.get_outputs()[0].name
    in_int8, out_int8 = session_int8.get_inputs()[0].name, session_int8.get_outputs()[0].name
    
    # Generate test image
    raw_img = Image.open("data/images/img_00001.jpg").convert('RGB').resize((224, 224))
    raw_np = np.array(raw_img).astype(np.float32)
    
    # Format 1: ImageNet Normalized (Mean/Std)
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    norm_img = ((raw_np / 255.0) - mean) / std
    t_norm = np.transpose(norm_img, (2, 0, 1))[np.newaxis, :].astype(np.float32)
    
    # Format 2: Normalized Float [0.0, 1.0]
    t_float_01 = np.transpose(raw_np / 255.0, (2, 0, 1))[np.newaxis, :].astype(np.float32)
    
    # Format 3: Raw Float [0.0, 255.0]
    t_float_255 = np.transpose(raw_np, (2, 0, 1))[np.newaxis, :].astype(np.float32)
    
    # Oracle Label from FP32
    oracle_res = session_fp32.run([out_fp32], {in_fp32: t_norm})[0]
    oracle_label = int(np.argmax(oracle_res))
    print(f"[+] Clean FP32 Oracle Prediction: Class #{oracle_label}")
    print("-" * 65)
    
    formats = [
        ("Format 1: ImageNet Normalized [-2.1, 2.6]", t_norm),
        ("Format 2: Float [0.0, 1.0]", t_float_01),
        ("Format 3: Float [0.0, 255.0]", t_float_255)
    ]
    
    best_format = None
    for name, tensor in formats:
        res = session_int8.run([out_int8], {in_int8: tensor})[0]
        pred = int(np.argmax(res))
        match = "MATCH (SUCCESS)" if pred == oracle_label else "MISMATCH"
        print(f"{name:<45} -> Pred: #{pred:<4} | {match}")
        if pred == oracle_label:
            best_format = name
            
    print("=" * 65)
    if best_format:
        print(f"[SUCCESS] Correct INT8 Input Format Identified: {best_format}")
    else:
        print("[!] Trying uint8 format...")
    print("=" * 65)

if __name__ == "__main__":
    diagnose()
