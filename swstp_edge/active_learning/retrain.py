"""
retrain.py — Active-Learning Retraining & Deployment Root Launcher
-------------------------------------------------------------------
Forwards execution to retrain/retrain.py using runpy to avoid name shadowing.
Allows running directly from swstp_edge/active_learning/:
    python retrain.py
"""

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TARGET = os.path.join(_HERE, "retrain", "retrain.py")

if __name__ == "__main__":
    runpy.run_path(_TARGET, run_name="__main__")
