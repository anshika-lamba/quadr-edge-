# Architecture: MobileNetV4-Conv-Small
- Input Tensor: 1x3x224x224 (NCHW, standard float32)
- Execution Target: ONNX Runtime (CPUExecutionProvider ARM64)
- Target Quantization: Static/Dynamic Symmetric INT8
- Benchmark Latency Target: < 15 ms/frame
- Memory Target (RSS): < 200 MB
