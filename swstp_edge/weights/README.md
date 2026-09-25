# weights/

This directory stores the production ONNX models executed by the SWSTP edge node, as well as the baseline PyTorch `.pt` checkpoint used by the active-learning retraining suite.

---

## Model Files Overview

| File | Format | Purpose | Role |
|---|---|---|---|
| `best_int8.onnx` | ONNX (INT8) | Primary runtime inference | Fastest on Pi 4B CPU (~2× speedup via ONNX Runtime) |
| `best.onnx` | ONNX (FP32) | Fallback runtime inference | Standard full-precision ONNX model |
| `best.pt` | PyTorch | Active-learning fine-tuning | Baseline model used by `retrain.py` for transfer learning |

---

## Active-Learning Model Replacement Policy

When active-learning retraining is performed using the accessory suite in `swstp_edge/active_learning/`:
1. `retrain.py` trains on a reviewed batch starting from `best.pt`.
2. The newly trained checkpoint is exported to ONNX format.
3. The newly produced ONNX file **replaces** the existing `best.onnx` (and optionally `best_int8.onnx`) in this folder.
4. An automatic timestamped backup of the previous production model (e.g. `best.onnx.bak_<timestamp>`) is created before replacement for easy rollback.
5. The edge runtime automatically resolves and uses the updated ONNX model on its next run.

---

## Manual ONNX Export from `best.pt`

```bash
# Using the active-learning export tool:
python ../active_learning/export/export_onnx.py --weights best.pt

# Or directly with Ultralytics CLI:
# INT8 quantised (fastest on Pi 4B CPU):
yolo export model=best.pt format=onnx int8=True

# FP32 (standard):
yolo export model=best.pt format=onnx
```

---

## Current Status
- `best.onnx` and `best_int8.onnx` are deployed for production inference.
- `best.pt` is present as the base training checkpoint.
- Inference backend selection: ONNX Runtime (if installed) → OpenCV DNN (fallback).
- Expected inference time on Pi 4B: ~300–600 ms per triggered frame (runs in background thread).
