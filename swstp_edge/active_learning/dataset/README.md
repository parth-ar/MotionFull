# Dataset Branch (`active_learning/dataset/`)

Organizes all images, labels, and YOLO format dataset splits used during the active-learning loop.

## Subdirectories

- `raw/`: Place incoming, unreviewed candidate frames here. Images can be `.jpg`, `.png`, `.jpeg`, or `.webp`.
- `reviewed/`: Destination directory for `review_app.py`. Stores individual batches:
  ```text
  reviewed/batch_YYYY_MM_DD/
  ├── images/              # Verified frame captures
  ├── labels/              # Corresponding YOLO format .txt files
  └── review_state.json    # Record of accepted and rejected filenames
  ```
- `splits/`: Staging area where `retrain.py` builds the train/val split datasets and generates `data.yaml`.

## Negative Samples (Background Frames)
Frames reviewed and accepted without any boxes produce an empty label file (`<filename>.txt`). Empty label files are valid negative samples in YOLO training and help teach the model to reduce false positives.
