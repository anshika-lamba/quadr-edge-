import os
import tracemalloc
import numpy as np
import onnxruntime as ort

def get_proc_rss_mib():
    """Reads Linux kernel VmRSS from /proc/self/status for native C++ memory."""
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0  # KiB to MiB
    except Exception:
        pass
    return 0.0

def profile_precise_memory():
    print("=" * 65)
    print("QADR Memory Profiling: Native Kernel VmRSS + Python tracemalloc")
    print("=" * 65)
    
    # 1. Base Python Interpreter Memory
    base_rss = get_proc_rss_mib()
    
    # 2. FP32 Model C++ Session Allocation
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session_fp32 = ort.InferenceSession("models/mobilenetv4_small_fp32.onnx", opts)
    fp32_rss = get_proc_rss_mib()
    fp32_allocated = fp32_rss - base_rss
    
    # 3. INT8 Model C++ Session Allocation
    session_int8 = ort.InferenceSession("models/mobilenetv4_small_int8.onnx", opts)
    int8_rss = get_proc_rss_mib()
    int8_allocated = int8_rss - fp32_rss
    
    # 4. QADR Pre-Op Working Set Memory (NumPy array allocation)
    tracemalloc.start()
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    _ = np.clip(dummy_input * 1.0376 + 0.0002, 0.0002, 0.9034)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    qadr_working_set_kb = peak_bytes / 1024.0
    
    print(f"Base Interpreter Memory        : {base_rss:.2f} MiB")
    print(f"FP32 Model Native Allocation   : {fp32_allocated:.2f} MiB")
    print(f"INT8 Model Native Allocation   : {int8_allocated:.2f} MiB")
    print(f"QADR Pre-Op Working Set Peak   : {qadr_working_set_kb:.2f} KiB ({qadr_working_set_kb/1024.0:.4f} MiB)")
    print("=" * 65)

if __name__ == "__main__":
    profile_precise_memory()
