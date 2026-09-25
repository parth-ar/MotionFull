"""
Active-Learning Fine-tune & Production ONNX Deployment
-------------------------------------------------------
Takes a reviewed batch from review_app.py (images/ + labels/, YOLO format,
including empty label files for confirmed-negative frames) and fine-tunes
an existing YOLO model on it.

This is intentionally conservative: low learning rate, few epochs, and an
optional frozen backbone — because you're correcting a model with a small
batch, not training from scratch. Since you don't have your original
dataset anymore, this also guards a bit against catastrophic forgetting
(the model unlearning things it used to know).

ONNX REPLACEMENT INTEGRATION:
  The edge node runs on ONNX weights (best.onnx / best_int8.onnx).
  Base weights (best.pt) live in swstp_edge/weights/.
  When training completes:
    1. The new best.pt checkpoint is produced.
    2. It is exported to ONNX format.
    3. The newly produced ONNX model replaces the old production ONNX
       weights (swstp_edge/weights/best.onnx) with an automatic backup
       saved for rollback safety.

Usage:
    python retrain.py \
        --batch-dir ../dataset/reviewed/batch_2026_09_25 \
        --weights ../../weights/best.pt \
        --classes "litter" \
        --output runs/run_01 \
        --replace-production
"""

import argparse
import datetime
import os
import random
import shutil
import sys
try:
    import yaml
except ImportError:
    yaml = None

# Ensure UTF-8 output on Windows consoles with Unicode usernames
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure swstp_edge and active_learning are in sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ACTIVE_LEARNING_DIR = os.path.abspath(os.path.join(_HERE, ".."))
_SWSTP_EDGE_DIR = os.path.abspath(os.path.join(_ACTIVE_LEARNING_DIR, ".."))
for p in (_SWSTP_EDGE_DIR, _ACTIVE_LEARNING_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


def resolve_default_paths():
    """Resolve default paths for batch dir, base weights, and output runs."""
    today_str = datetime.date.today().strftime("%Y_%m_%d")

    # Look for most recent batch in reviewed dir
    reviewed_dir = os.path.join(_ACTIVE_LEARNING_DIR, "dataset", "reviewed")
    default_batch = os.path.join(reviewed_dir, f"batch_{today_str}")

    if not os.path.isdir(default_batch) and os.path.isdir(reviewed_dir):
        batches = sorted([
            os.path.join(reviewed_dir, d) for d in os.listdir(reviewed_dir)
            if os.path.isdir(os.path.join(reviewed_dir, d))
        ])
        if batches:
            default_batch = batches[-1]

    # Weights candidate
    default_weights = os.path.join(_SWSTP_EDGE_DIR, "weights", "best.pt")
    if not os.path.isfile(default_weights):
        default_weights = os.path.join("weights", "best.pt")

    # Output run candidate
    default_output = os.path.join(_ACTIVE_LEARNING_DIR, "retrain", "runs", f"run_{today_str}")

    return default_batch, default_weights, default_output


def parse_args():
    default_batch, default_weights, default_output = resolve_default_paths()

    p = argparse.ArgumentParser(
        description="Fine-tune YOLO on a reviewed active-learning batch and deploy updated ONNX weights"
    )
    p.add_argument("--batch-dir", default=default_batch,
                   help="Output folder from review_app.py (contains images/ and labels/) (default: dataset/reviewed/batch_YYYY_MM_DD)")
    p.add_argument("--weights", default=default_weights,
                   help="Existing base .pt to fine-tune from (default: swstp_edge/weights/best.pt)")
    p.add_argument("--classes", default="litter",
                   help="Comma-separated class names, SAME ORDER as review_app.py used (default: litter)")
    p.add_argument("--output", default=default_output,
                   help="Folder for built dataset + training run (default: retrain/runs/run_YYYY_MM_DD)")
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction held out for validation (default: 0.2)")
    p.add_argument("--epochs", type=int, default=15,
                   help="Epochs for correction batch (keep low, default: 15)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Image size for YOLO training (default: 640)")
    p.add_argument("--lr0", type=float, default=0.001,
                   help="Initial learning rate (low LR to nudge, default: 0.001)")
    p.add_argument("--freeze", type=int, default=10,
                   help="Freeze this many backbone layers (0 to disable, default: 10)")
    p.add_argument("--device", default=None,
                   help="'cpu', '0' for first CUDA GPU, or leave unset for auto-detect")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for train/val split (default: 42)")

    # ONNX export and production replacement flags
    p.add_argument("--export-onnx", action="store_true", default=True,
                   help="Automatically export the retrained model to ONNX format (default: True)")
    p.add_argument("--no-export-onnx", dest="export_onnx", action="store_false",
                   help="Skip automatic ONNX export")
    p.add_argument("--replace-production", action="store_true", default=True,
                   help="Replace the old production ONNX model in weights/ with the new ONNX model (default: True)")
    p.add_argument("--no-replace-production", dest="replace_production", action="store_false",
                   help="Do not replace production ONNX weights automatically")
    p.add_argument("--backup-old", action="store_true", default=True,
                   help="Create a timestamped backup of old ONNX weights before replacement (default: True)")
    p.add_argument("--int8", action="store_true", default=False,
                   help="Also export an INT8 quantised ONNX model")

    return p.parse_args()


def build_dataset(batch_dir, output_dir, class_names, val_split, seed):
    images_dir = os.path.join(batch_dir, "images")
    labels_dir = os.path.join(batch_dir, "labels")
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        sys.exit(f"Expected {images_dir} and {labels_dir} to exist (output of review_app.py)")

    files = sorted(f for f in os.listdir(images_dir)
                   if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if not files:
        sys.exit(f"No images found in {images_dir}")

    random.Random(seed).shuffle(files)
    n_val = max(1, int(len(files) * val_split))
    val_files = set(files[:n_val])
    train_files = [f for f in files if f not in val_files]

    dataset_dir = os.path.join(output_dir, "dataset")
    for split, split_files in (("train", train_files), ("val", sorted(val_files))):
        img_out = os.path.join(dataset_dir, "images", split)
        lbl_out = os.path.join(dataset_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for f in split_files:
            stem = os.path.splitext(f)[0]
            shutil.copy2(os.path.join(images_dir, f), os.path.join(img_out, f))
            label_src = os.path.join(labels_dir, stem + ".txt")
            label_dst = os.path.join(lbl_out, stem + ".txt")
            if os.path.isfile(label_src):
                shutil.copy2(label_src, label_dst)
            else:
                # Confirmed-negative frame: empty label file is valid YOLO input.
                open(label_dst, "w", encoding="utf-8").close()

    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    if yaml is not None:
        data_yaml = {
            "path": os.path.abspath(dataset_dir),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(class_names)},
        }
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml, f, sort_keys=False)
    else:
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            f.write(f"path: {os.path.abspath(dataset_dir)}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("names:\n")
            for i, name in enumerate(class_names):
                f.write(f"  {i}: {name}\n")

    print(f"Dataset built: {len(train_files)} train / {len(val_files)} val -> {dataset_dir}")
    return data_yaml_path


def export_and_replace_onnx(model, new_weights_path, data_yaml_path, imgsz, replace_production, backup_old, int8_export):
    """
    Export the fine-tuned model to ONNX format and replace the production ONNX
    files in swstp_edge/weights/ as specified by configuration.
    """
    print("\n" + "=" * 65)
    print("EXPORTING FINE-TUNED MODEL TO ONNX FORMAT...")
    print("=" * 65)

    try:
        exported_path = model.export(format="onnx", imgsz=imgsz, dynamic=False)
        print(f"[EXPORT] Successfully exported standard ONNX model: {exported_path}")
    except Exception as e:
        print(f"[EXPORT ERROR] Failed to export ONNX directly from model instance: {e}")
        # Try loading fresh from new_weights_path
        from ultralytics import YOLO
        export_model = YOLO(new_weights_path)
        exported_path = export_model.export(format="onnx", imgsz=imgsz, dynamic=False)
        print(f"[EXPORT] Fallback export succeeded: {exported_path}")

    int8_exported_path = None
    if int8_export:
        try:
            print("[EXPORT] Exporting INT8 quantized ONNX model...")
            from ultralytics import YOLO
            export_model = YOLO(new_weights_path)
            int8_exported_path = export_model.export(format="onnx", imgsz=imgsz, int8=True, data=data_yaml_path)
            print(f"[EXPORT] Successfully exported INT8 ONNX model: {int8_exported_path}")
        except Exception as e:
            print(f"[EXPORT WARNING] INT8 quantization failed ({e}). Proceeding with standard ONNX.")

    if not replace_production:
        print("\n[NOTE] --no-replace-production was set. Production ONNX files were NOT modified.")
        print(f"Newly produced ONNX file is located at: {exported_path}")
        return exported_path

    # Production weights target paths in swstp_edge/weights/
    prod_weights_dir = os.path.join(_SWSTP_EDGE_DIR, "weights")
    os.makedirs(prod_weights_dir, exist_ok=True)
    target_prod_onnx = os.path.join(prod_weights_dir, "best.onnx")
    target_prod_int8 = os.path.join(prod_weights_dir, "best_int8.onnx")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Replace standard best.onnx
    if os.path.isfile(target_prod_onnx) and backup_old:
        backup_path = f"{target_prod_onnx}.bak_{timestamp}"
        shutil.copy2(target_prod_onnx, backup_path)
        print(f"[BACKUP] Existing production model backed up to: {backup_path}")

    shutil.copy2(exported_path, target_prod_onnx)
    print(f"\n[PRODUCTION REPLACEMENT SUCCESS]")
    print(f" -> Newly produced ONNX model replaced old production file at:")
    print(f"    {target_prod_onnx}")

    # 2. Replace INT8 if available
    if int8_exported_path and os.path.isfile(int8_exported_path):
        if os.path.isfile(target_prod_int8) and backup_old:
            backup_int8_path = f"{target_prod_int8}.bak_{timestamp}"
            shutil.copy2(target_prod_int8, backup_int8_path)
            print(f"[BACKUP] Existing INT8 model backed up to: {backup_int8_path}")
        shutil.copy2(int8_exported_path, target_prod_int8)
        print(f" -> Newly produced INT8 ONNX model replaced old file at:")
        print(f"    {target_prod_int8}")

    return exported_path


def main():
    args = parse_args()
    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not class_names:
        sys.exit("--classes must list at least one class name")

    # Ensure weights path exists
    if not os.path.isfile(args.weights):
        # Check swstp_edge/weights/best.pt
        alt_weights = os.path.join(_SWSTP_EDGE_DIR, "weights", "best.pt")
        if os.path.isfile(alt_weights):
            args.weights = alt_weights
        else:
            sys.exit(f"Base weights not found: {args.weights}")

    data_yaml_path = build_dataset(args.batch_dir, args.output, class_names, args.val_split, args.seed)

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "Ultralytics is required for retraining. "
            "Please install dependencies with: pip install ultralytics torch torchvision"
        )

    print(f"Loading base weights: {args.weights}")
    model = YOLO(args.weights)

    train_kwargs = dict(
        data=data_yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        lr0=args.lr0,
        project=os.path.abspath(args.output),
        name="finetune",
        exist_ok=True,
    )
    if args.freeze > 0:
        train_kwargs["freeze"] = args.freeze
    if args.device is not None:
        train_kwargs["device"] = args.device

    print(f"\nFine-tuning model: epochs={args.epochs}, lr0={args.lr0}, freeze={args.freeze}")
    results = model.train(**train_kwargs)

    run_dir = os.path.join(os.path.abspath(args.output), "finetune")
    new_weights = os.path.join(run_dir, "weights", "best.pt")

    print("\n" + "=" * 65)
    print(f"Training complete. Newly trained weights: {new_weights}")
    print("=" * 65)

    if args.export_onnx and os.path.isfile(new_weights):
        export_and_replace_onnx(
            model=model,
            new_weights_path=new_weights,
            data_yaml_path=data_yaml_path,
            imgsz=args.imgsz,
            replace_production=args.replace_production,
            backup_old=args.backup_old,
            int8_export=args.int8,
        )

    print("\n" + "=" * 65)
    print("ACTIVE-LEARNING WORKFLOW SUMMARY:")
    print(f" - Reviewed batch:   {args.batch_dir}")
    print(f" - Train/Val split:   {data_yaml_path}")
    print(f" - PyTorch Weights:   {new_weights}")
    if args.replace_production:
        print(f" - Production ONNX:   {os.path.join(_SWSTP_EDGE_DIR, 'weights', 'best.onnx')} (REPLACED)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
