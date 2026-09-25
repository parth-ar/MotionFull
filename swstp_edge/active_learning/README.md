# Active-Learning & Model Retraining Accessory

A dedicated, offline accessory suite for curating mini active-learning datasets, fine-tuning YOLO litter detection models, and automatically deploying the newly produced ONNX models to the SWSTP edge node.

> **Important**: This folder is an **accessory tool**. It is completely decoupled from and does **NOT** run with the primary edge node daemon (`swstp-unified.service` / `main.py`).

---

## Architecture & Organization Branches

```text
swstp_edge/active_learning/
├── config.py                  # Master active-learning & deployment configuration
├── requirements-accessory.txt # Standalone dependencies (PyTorch, Ultralytics, ONNX)
├── review_app.py              # Root launcher for review application
├── retrain.py                 # Root launcher for retraining & ONNX replacement
│
├── review/                    # Branch: Annotation, bbox correction & review
│   ├── review_app.py          # Interactive OpenCV bounding box review tool
│   └── README.md              # Review controls & workflow guide
│
├── retrain/                   # Branch: YOLO fine-tuning & training runs
│   ├── retrain.py             # Conservative active-learning fine-tune engine
│   ├── runs/                  # Checkpoints, validation curves, and logs
│   └── README.md              # Retraining hyperparameter guide
│
├── dataset/                   # Branch: Dataset staging & YOLO splits
│   ├── raw/                   # Incoming frames / captured images to review
│   ├── reviewed/              # Confirmed batches (images/ + labels/ + state)
│   ├── splits/                # Auto-generated train/val splits + data.yaml
│   └── README.md              # Dataset format specifications
│
└── export/                    # Branch: ONNX conversion & model deployment
    ├── export_onnx.py         # PyTorch (.pt) -> ONNX exporter & deployer
    └── README.md              # Production ONNX replacement details
```

---

## Model Format & Replacement Policy

- **Edge Runtime Format**: The live edge detector exclusively uses `.onnx` models (`swstp_edge/weights/best.onnx` and `best_int8.onnx`) for high performance and low CPU overhead on Raspberry Pi 4B.
- **Base PyTorch Weights**: The original format weights (`best.pt`) are located in [`swstp_edge/weights/best.pt`](../weights/best.pt) and serve as the base checkpoint for transfer learning.
- **Automatic Replacement**: When `retrain.py` finishes:
  1. Fine-tuned PyTorch weights (`best.pt`) are produced in the run directory.
  2. The model is automatically exported to ONNX format.
  3. The newly produced `.onnx` file **replaces** the old production ONNX file ([`swstp_edge/weights/best.onnx`](../weights/best.onnx)).
  4. An automatic timestamped backup of the previous model (`best.onnx.bak_<timestamp>`) is created for rollback security.

---

## Complete Active-Learning Lifecycle

### Step 1: Collect & Review Frames
Place raw test or triggered frames in `dataset/raw/`, then launch the review app:

```bash
# From swstp_edge/active_learning/:
python review_app.py
```
- Draw, move, resize, or delete boxes using your mouse.
- Press `s` or `Enter` to accept a frame.
- Accepted frames and YOLO labels are written to `dataset/reviewed/batch_<YYYY_MM_DD>/`.

### Step 2: Fine-Tune the Model & Replace Production ONNX
Run the fine-tuning script:

```bash
# Fine-tune and automatically replace production ONNX weights:
python retrain.py --batch-dir dataset/reviewed/batch_2026_09_25
```
- Splits the batch into train/val sets.
- Conservatively fine-tunes with a low learning rate (`lr0=0.001`) and frozen backbone layers (`freeze=10`) to prevent catastrophic forgetting.
- Exports the fine-tuned model to `.onnx`.
- Overwrites [`swstp_edge/weights/best.onnx`](../weights/best.onnx) with the new model.

### Step 3 (Optional): Standalone ONNX Export
If you want to manually convert any `.pt` weight and deploy it:

```bash
python export/export_onnx.py --weights retrain/runs/run_2026_09_25/finetune/weights/best.pt
```
