# Export Branch (`active_learning/export/`)

Tools for exporting PyTorch `.pt` models to ONNX and deploying them into the production `swstp_edge/weights/` folder.

## Scripts

### `export_onnx.py`
Exports a trained PyTorch checkpoint (`.pt`) to ONNX and replaces the live model:

```bash
# Export and replace production best.onnx:
python export_onnx.py --weights ../retrain/runs/run_01/finetune/weights/best.pt

# Export only (do not touch swstp_edge/weights/):
python export_onnx.py --weights ../retrain/runs/run_01/finetune/weights/best.pt --no-replace

# Export both FP32 and INT8 ONNX:
python export_onnx.py --weights ../../weights/best.pt --int8 --data ../dataset/splits/data.yaml
```

## Production Safety
- Whenever a production ONNX model is replaced, the tool automatically makes a backup:
  `swstp_edge/weights/best.onnx.bak_<timestamp>`
- If needed, you can easily restore a prior model by renaming the `.bak` file back to `best.onnx`.
