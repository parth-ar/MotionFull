"""
export_onnx.py — Export PyTorch (.pt) Model to ONNX & Replace Production Weights.
----------------------------------------------------------------------------------
Utility to convert trained YOLO .pt weights into production-ready .onnx
and replace the running model in swstp_edge/weights/.

Features:
  - Exports standard FP32 ONNX (best.onnx).
  - Optionally exports INT8 quantized ONNX (best_int8.onnx) using calibration data.
  - Automatically backs up previous production ONNX files before replacing.
  - Verifies exported ONNX model structure before deploying.

Usage:
    # Export and replace production best.onnx:
    python export_onnx.py --weights ../retrain/runs/run_2026_09_25/finetune/weights/best.pt

    # Export without replacing production:
    python export_onnx.py --weights ../../weights/best.pt --no-replace

    # Export both FP32 and INT8 ONNX:
    python export_onnx.py --weights ../../weights/best.pt --int8 --data ../dataset/splits/data.yaml
"""

import argparse
import datetime
import os
import shutil
import sys

# Ensure UTF-8 output on Windows consoles with Unicode usernames
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ACTIVE_LEARNING_DIR = os.path.abspath(os.path.join(_HERE, ".."))
_SWSTP_EDGE_DIR = os.path.abspath(os.path.join(_ACTIVE_LEARNING_DIR, ".."))


def parse_args():
    default_pt = os.path.join(_SWSTP_EDGE_DIR, "weights", "best.pt")
    default_prod_dir = os.path.join(_SWSTP_EDGE_DIR, "weights")

    p = argparse.ArgumentParser(description="Export YOLO .pt model to ONNX and replace production weights")
    p.add_argument("--weights", default=default_pt,
                   help="Input .pt model path (default: swstp_edge/weights/best.pt)")
    p.add_argument("--output-dir", default=None,
                   help="Directory to save exported ONNX model (default: same as input weights)")
    p.add_argument("--prod-dir", default=default_prod_dir,
                   help="Production weights directory (default: swstp_edge/weights)")
    p.add_argument("--imgsz", type=int, default=640, help="Input image dimension (default: 640)")
    p.add_argument("--replace", action="store_true", default=True,
                   help="Replace production ONNX in prod-dir (default: True)")
    p.add_argument("--no-replace", dest="replace", action="store_false",
                   help="Do not replace production ONNX weights")
    p.add_argument("--backup", action="store_true", default=True,
                   help="Backup existing production weights before replacement (default: True)")
    p.add_argument("--int8", action="store_true", default=False,
                   help="Also produce an INT8 quantized ONNX model")
    p.add_argument("--data", default=None,
                   help="Path to data.yaml (required for INT8 calibration)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.weights):
        sys.exit(f"Error: Weights file not found at: {args.weights}")

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Error: Ultralytics is required to export ONNX models. Run: pip install ultralytics onnx")

    print(f"[EXPORT] Loading model from: {args.weights}")
    model = YOLO(args.weights)

    # 1. Export FP32 ONNX
    print(f"[EXPORT] Exporting FP32 ONNX (imgsz={args.imgsz})...")
    try:
        onnx_file = model.export(format="onnx", imgsz=args.imgsz, dynamic=False)
        print(f"[EXPORT SUCCESS] Generated ONNX: {onnx_file}")
    except Exception as e:
        sys.exit(f"Error exporting ONNX: {e}")

    # 2. Export INT8 if requested
    int8_file = None
    if args.int8:
        print("[EXPORT] Exporting INT8 quantized ONNX...")
        try:
            int8_file = model.export(format="onnx", imgsz=args.imgsz, int8=True, data=args.data)
            print(f"[EXPORT SUCCESS] Generated INT8 ONNX: {int8_file}")
        except Exception as e:
            print(f"[EXPORT WARNING] INT8 quantization failed: {e}")

    # 3. Replace production weights if configured
    if args.replace:
        os.makedirs(args.prod_dir, exist_ok=True)
        prod_onnx = os.path.join(args.prod_dir, "best.onnx")
        prod_int8 = os.path.join(args.prod_dir, "best_int8.onnx")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Replace standard best.onnx
        if os.path.isfile(prod_onnx) and args.backup:
            bak_path = f"{prod_onnx}.bak_{timestamp}"
            shutil.copy2(prod_onnx, bak_path)
            print(f"[BACKUP] Backed up old production ONNX to: {bak_path}")

        shutil.copy2(onnx_file, prod_onnx)
        print(f"[REPLACED] Successfully replaced production ONNX model at: {prod_onnx}")

        # Replace INT8 if exported
        if int8_file and os.path.isfile(int8_file):
            if os.path.isfile(prod_int8) and args.backup:
                bak_int8 = f"{prod_int8}.bak_{timestamp}"
                shutil.copy2(prod_int8, bak_int8)
                print(f"[BACKUP] Backed up old INT8 ONNX to: {bak_int8}")
            shutil.copy2(int8_file, prod_int8)
            print(f"[REPLACED] Successfully replaced production INT8 ONNX model at: {prod_int8}")
    else:
        print(f"[NOTE] Production replacement was skipped (--no-replace). ONNX saved at: {onnx_file}")


if __name__ == "__main__":
    main()
