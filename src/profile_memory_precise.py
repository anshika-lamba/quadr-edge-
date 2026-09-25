import tracemalloc
import time
import numpy as np
import onnxruntime as ort

def profile_precise_memory():
    print("=" * 65)
    print("QADR Fix 2: Precise Memory Allocation Profiling (tracemalloc)")
    print("=" * 65)
    
    tracemalloc.start()
    
    # 1. Base Python Interpreter Memory
    current, peak = tracemalloc.get_traced_memory()
    base_mem_mb = peak / (1024 * 1024)
    
    # 2. Session Load FP32 Memory
    tracemalloc.reset_peak()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session_fp32 = ort.InferenceSession("models/mobilenetv4_small_fp32.onnx", opts)
    current, peak = tracemalloc.get_traced_memory()
    fp32_mem_mb = peak / (1024 * 1024)
    
    # 3. Session Load INT8 Memory
    tracemalloc.reset_peak()
    session_int8 = ort.InferenceSession("models/mobilenetv4_small_int8.onnx", opts)
    current, peak = tracemalloc.get_traced_memory()
    int8_mem_mb = peak / (1024 * 1024)
    
    # 4. QADR Pre-Op Working Set Memory
    tracemalloc.reset_peak()
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    _ = np.clip(dummy_input * 1.0376 + 0.0002, 0.0002, 0.9034)
    current, peak = tracemalloc.get_traced_memory()
    qadr_mem_kb = peak / 1024.0
    
    print(f"Base Python Interpreter Memory : {base_mem_mb:.2f} MiB")
    print(f"FP32 Model Allocated Memory    : {fp32_mem_mb:.2f} MiB")
    print(f"INT8 Model Allocated Memory    : {int8_mem_mb:.2f} MiB")
    print(f"QADR Pre-Op Working Set Peak   : {qadr_mem_kb:.2f} KiB ({qadr_mem_kb/1024.0:.4f} MiB)")
    print("=" * 65)

if __name__ == "__main__":
    profile_precise_memory()
