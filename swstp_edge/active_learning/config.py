"""
config.py — Central Configuration for Active-Learning & Retraining Accessory.
------------------------------------------------------------------------------
Organizes paths, dataset branches, class mappings, base PyTorch weights,
and the production ONNX model replacement policy.

NOTE: This active-learning suite is an offline accessory tool and does NOT
run with the primary swstp-unified edge runtime service.
"""

import os

# ---------------------------------------------------------------------------
# Path Anchors
# ---------------------------------------------------------------------------
# Current directory: swstp_edge/active_learning/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# swstp_edge project directory
SWSTP_EDGE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Production weights directory (swstp_edge/weights/)
WEIGHTS_DIR = os.path.join(SWSTP_EDGE_DIR, "weights")

# Base PyTorch weights (.pt) used for fine-tuning
BASE_PT_WEIGHTS = os.path.join(WEIGHTS_DIR, "best.pt")

# Production ONNX models used by the edge runtime
PROD_ONNX_PATH = os.path.join(WEIGHTS_DIR, "best.onnx")
PROD_INT8_ONNX_PATH = os.path.join(WEIGHTS_DIR, "best_int8.onnx")

# ---------------------------------------------------------------------------
# Dataset Branch Paths (swstp_edge/active_learning/dataset/)
# ---------------------------------------------------------------------------
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RAW_IMAGES_DIR = os.path.join(DATASET_DIR, "raw")
REVIEWED_DIR = os.path.join(DATASET_DIR, "reviewed")
SPLITS_DIR = os.path.join(DATASET_DIR, "splits")

# Retraining runs output directory (swstp_edge/active_learning/retrain/runs/)
RETRAIN_RUNS_DIR = os.path.join(BASE_DIR, "retrain", "runs")

# ---------------------------------------------------------------------------
# Model Classes
# ---------------------------------------------------------------------------
DEFAULT_CLASSES = "litter"
# For multi-class pipelines, supply a comma-separated string, e.g.:
# "cigarette,plastic_bottle,wrapper,can,other"

# ---------------------------------------------------------------------------
# Active-Learning Production Replacement Policy
# ---------------------------------------------------------------------------
# The main edge service (swstp_edge) exclusively runs ONNX models for high
# efficiency on the Pi 4B CPU.
#
# When fine-tuning finishes in retrain.py:
#   1. A new best.pt checkpoint is produced.
#   2. The model is exported to ONNX format (best.onnx).
#   3. If REPLACE_PRODUCTION_ONNX is True, the newly produced ONNX file
#      replaces the old production ONNX file (swstp_edge/weights/best.onnx).
#   4. If BACKUP_OLD_ONNX is True, a timestamped backup of the previous
#      ONNX model is preserved before replacement.
AUTO_REPLACE_PRODUCTION_ONNX = True
BACKUP_OLD_ONNX = True
