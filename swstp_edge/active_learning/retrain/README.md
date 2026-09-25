# Retrain Branch (`active_learning/retrain/`)

Fine-tunes the baseline YOLO model (`best.pt`) on reviewed batches and exports the new model to replace old production ONNX weights.

## Retraining Philosophy: Conservative Fine-Tuning
When fine-tuning on a small active-learning batch (e.g. 50–200 images):
- **Low Learning Rate (`--lr0 0.001`)**: Gently nudges weights without blowing up existing learned features.
- **Frozen Backbone (`--freeze 10`)**: Freezes earlier feature-extraction layers so the model preserves foundational edge/texture representations and avoids catastrophic forgetting.
- **Short Epochs (`--epochs 15`)**: Prevents overfitting to the small batch.

## ONNX Production Replacement
By default, `--replace-production` is **enabled**. When training finishes:
1. `model.export(format="onnx")` generates a new `best.onnx`.
2. Existing `swstp_edge/weights/best.onnx` is backed up to `best.onnx.bak_<timestamp>`.
3. The new `best.onnx` is copied into `swstp_edge/weights/best.onnx`.

## CLI Usage

```bash
# Standard retraining with automatic ONNX export & production replacement:
python retrain.py --batch-dir ../dataset/reviewed/batch_2026_09_25

# Train without replacing production weights immediately:
python retrain.py --batch-dir ../dataset/reviewed/batch_2026_09_25 --no-replace-production

# Retrain on a specific CUDA GPU with custom epochs:
python retrain.py --batch-dir ../dataset/reviewed/batch_01 --epochs 20 --device 0
```
