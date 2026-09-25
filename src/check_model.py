import numpy as np
import onnxruntime as ort
from PIL import Image

def check_model_nodes():
    print("=" * 65)
    print("QADR Diagnostics: Real Image Inspection (img_00001.jpg)")
    print("=" * 65)
    
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session_fp32 = ort.InferenceSession("models/mobilenetv4_small_fp32.onnx", opts)
    session_int8 = ort.InferenceSession("models/mobilenetv4_small_int8.onnx", opts)
    
    in_fp32 = session_fp32.get_inputs()[0]
    in_int8 = session_int8.get_inputs()[0]
    
    print(f"[FP32] Node Name: {in_fp32.name} | Type: {in_fp32.type}")
    print(f"[INT8] Node Name: {in_int8.name} | Type: {in_int8.type}")
    print("-" * 65)
    
    # Load real photo
    img = Image.open("data/images/img_00001.jpg").convert('RGB').resize((224, 224))
    img_arr = np.array(img).astype(np.float32) / 255.0
    
    # ImageNet Normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (img_arr - mean) / std
    tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, :].astype(np.float32)
    
    out_fp32_name = session_fp32.get_outputs()[0].name
    out_int8_name = session_int8.get_outputs()[0].name
    
    logits_fp32 = session_fp32.run([out_fp32_name], {in_fp32.name: tensor})[0][0]
    logits_int8 = session_int8.run([out_int8_name], {in_int8.name: tensor})[0][0]
    
    top5_fp32 = np.argsort(logits_fp32)[-5:][::-1]
    top5_int8 = np.argsort(logits_int8)[-5:][::-1]
    
    top1_fp32 = top5_fp32[0]
    top1_int8 = top5_int8[0]
    
    print(f"FP32 Top-1 Prediction: Class #{top1_fp32}")
    print(f"INT8 Top-1 Prediction: Class #{top1_int8}")
    print(f"FP32 Top-5 Predicted Classes: {top5_fp32}")
    print(f"INT8 Top-5 Predicted Classes: {top5_int8}")
    
    overlap = len(set(top5_fp32).intersection(set(top5_int8)))
    print(f"[+] Top-5 Overlap: {overlap}/5 classes match!")
    
    match_str = "SUCCESS (Top-1 Match)" if top1_fp32 == top1_int8 else "Top-1 Mismatch"
    print(f"[STATUS] {match_str}")
    print("=" * 65)

if __name__ == "__main__":
    check_model_nodes()
