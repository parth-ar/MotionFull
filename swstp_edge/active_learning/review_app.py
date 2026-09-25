"""
review_app.py — Active-Learning Review Application Root Launcher
-----------------------------------------------------------------
Forwards execution to review/review_app.py using runpy to avoid name shadowing.
Allows running directly from swstp_edge/active_learning/:
    python review_app.py
"""

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TARGET = os.path.join(_HERE, "review", "review_app.py")

if __name__ == "__main__":
    runpy.run_path(_TARGET, run_name="__main__")
