# weights/

Place YOLO ONNX model weights here.

## Required files (in order of preference):

1. `best_int8.onnx` — INT8 quantised ONNX model (recommended for Pi 4B, ~2× faster)
2. `best.onnx`      — FP32 ONNX model (fallback if int8 not available)

## How to generate from an Ultralytics .pt file:

```bash
# INT8 quantised (fastest on Pi 4B CPU):
yolo export model=best.pt format=onnx int8=True

# FP32 (standard):
yolo export model=best.pt format=onnx
```

## Current status:
- `best.onnx` and `best_int8.onnx` have been copied from the Litter-Detection-main project.
- Inference backend selection: ONNX Runtime (if installed) → OpenCV DNN (fallback).
- Expected inference time on Pi 4B: ~300–600 ms per triggered frame (runs in background thread).
