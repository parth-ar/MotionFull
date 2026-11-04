"""
test_roi_side_classifier.py
Interactive & automated test harness for ROI geometric left/right dumping side classification.

Usage:
  python test_roi_side_classifier.py                          # Live camera 0 (default)
  python test_roi_side_classifier.py --source 1               # Live camera 1
  python test_roi_side_classifier.py --source clip.mp4        # Video file
  python test_roi_side_classifier.py --roi custom_roi.json    # Load custom ROI JSON
  python test_roi_side_classifier.py --roi-box 0.1,0.2,0.9,0.8 # Custom ROI box (norm coords)
  python test_roi_side_classifier.py --static                 # Synthetic test (no camera needed)
  python test_roi_side_classifier.py --headless               # Headless test suite & report

Interactive Custom ROI Controls (Live Window):
  r      - Toggle Polygon Draw Mode:
             • Left-click  : Add vertex
             • Right-click : Undo last vertex
             • Click P1 / Enter / Space / r : Finish & apply polygon
  d      - Toggle Drag-ROI Mode:
             • Left-click & drag a rectangle on the video feed to set custom ROI instantly
  w      - Write / Save active ROI to roi_polygon.json
  l      - Reload ROI from disk (roi_polygon.json)
  c      - Clear in-progress polygon vertices
  1..4   - Quick Presets:
             [1] Full Frame (5% - 95%)
             [2] Road Center (20% - 80% X, 20% - 85% Y)
             [3] Lower Road / Verge (5% - 95% X, 40% - 95% Y)
             [4] Right Roadside Verge (55% - 95% X, 25% - 85% Y)
  b      - Toggle background subtraction motion detection
  s      - Save snapshot (.jpg) + sidecar (.json) to test_captures/
  h      - Toggle keyboard shortcuts help overlay
  q/ESC  - Exit
"""

from __future__ import annotations
import argparse
import datetime
import json
import os
import sys
import time

import cv2
import imutils
import numpy as np

# Ensure UTF-8 output on Windows consoles with non-ASCII username paths
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure local imports work cleanly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from roi_side_classifier import (
    classify_contour_side,
    classify_bbox_side,
    draw_side_split_overlay,
)

# ---------------------------------------------------------------------------
# Default parameters & paths
# ---------------------------------------------------------------------------
DEFAULT_ROI_FILE = os.path.join(SCRIPT_DIR, "roi_polygon.json")
SAVE_DIR         = os.path.join(SCRIPT_DIR, "test_captures")
DIFF_THRESHOLD   = 22
MIN_CONTOUR_AREA = 350
BG_ALPHA         = 0.04
WARMUP_FRAMES    = 25
PROC_SIZE        = (640, 360)  # Downscaled processing size (w, h)
WIN_TITLE        = "ROI Dumping Side Classifier - Test Harness"

PRESET_ROIS = {
    "1": ("Full Frame", [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]]),
    "2": ("Road Center", [[0.20, 0.20], [0.80, 0.20], [0.80, 0.85], [0.20, 0.85]]),
    "3": ("Lower Verge", [[0.05, 0.40], [0.95, 0.40], [0.95, 0.95], [0.05, 0.95]]),
    "4": ("Right Verge", [[0.55, 0.25], [0.95, 0.25], [0.95, 0.85], [0.55, 0.85]]),
}


def parse_roi_input(roi_str: str | None, roi_box_str: str | None, default_file: str = DEFAULT_ROI_FILE) -> list:
    """Parse custom ROI from file path, JSON string, comma-separated box, or default file."""
    # 1. Direct bounding box parameter: "x1,y1,x2,y2"
    if roi_box_str:
        try:
            parts = [float(p.strip()) for p in roi_box_str.split(",")]
            if len(parts) == 4:
                x1, y1, x2, y2 = parts
                print(f"[ROI] Parsed custom box: ({x1}, {y1}) -> ({x2}, {y2})")
                return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        except Exception as e:
            print(f"[ROI] Failed to parse --roi-box '{roi_box_str}': {e}")

    # 2. Path to custom JSON file or raw JSON string
    if roi_str:
        if os.path.isfile(roi_str):
            try:
                with open(roi_str, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) >= 3:
                    print(f"[ROI] Loaded custom ROI ({len(data)} vertices) from {roi_str}")
                    return data
            except Exception as e:
                print(f"[ROI] Error loading custom file '{roi_str}': {e}")
        else:
            # Try parsing as JSON string directly: [[x,y],...]
            try:
                data = json.loads(roi_str)
                if isinstance(data, list) and len(data) >= 3:
                    print(f"[ROI] Parsed custom JSON array ({len(data)} vertices)")
                    return data
            except Exception:
                pass

    # 3. Default roi_polygon.json
    if os.path.isfile(default_file):
        try:
            with open(default_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) >= 3:
                print(f"[ROI] Loaded active ROI ({len(data)} vertices) from {default_file}")
                return data
        except Exception as e:
            print(f"[ROI] Warning loading {default_file}: {e}")

    # 4. Fallback default
    print("[ROI] Using fallback Road Center ROI.")
    return [list(p) for p in PRESET_ROIS["2"][1]]


def save_roi_to_file(pts: list, target_file: str = DEFAULT_ROI_FILE) -> bool:
    """Save polygon ROI vertices to JSON file."""
    try:
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(pts, f, indent=2)
        print(f"[ROI] Saved custom ROI ({len(pts)} vertices) to {target_file}")
        return True
    except Exception as e:
        print(f"[ROI] Error saving to {target_file}: {e}")
        return False


# ---------------------------------------------------------------------------
# Global Interactive State
# ---------------------------------------------------------------------------
class TestState:
    def __init__(self, roi_polygon: list, roi_path: str = DEFAULT_ROI_FILE):
        self.roi_pts = roi_polygon
        self.roi_path = roi_path

        # Polygon Draw Mode ('r')
        self.is_drawing_poly = False
        self.drawn_pts: list = []

        # Drag Rectangle ROI Mode ('d')
        self.is_dragging_roi = False
        self.roi_drag_start = None
        self.roi_drag_preview = None  # (x1, y1, x2, y2)

        # Manual test bbox (Right-click drag)
        self.test_box_drag_start = None
        self.manual_bbox = None  # (x1, y1, x2, y2)

        # Motion & Classification State
        self.bg_model = None
        self.frame_count = 0
        self.motion_enabled = True
        self.show_help = False
        self.last_side_result = None
        self.snap_count = 0

        # On-screen Toast Notification
        self.toast_msg = ""
        self.toast_until = 0.0
        self.toast_col = (0, 255, 120)

    def show_toast(self, msg: str, duration: float = 3.2, col=(0, 255, 120)):
        self.toast_msg = msg
        self.toast_until = time.time() + duration
        self.toast_col = col

    def reset_motion(self):
        self.bg_model = None
        self.frame_count = 0


# ---------------------------------------------------------------------------
# Mouse Callback
# ---------------------------------------------------------------------------
def make_mouse_callback(state: TestState, win_w: int, win_h: int):
    def on_mouse(event, x, y, flags, param):
        w = param.get("w", win_w)
        h = param.get("h", win_h)

        # -------------------------------------------------------------------
        # Mode 1: Polygon Drawing Mode ('r')
        # -------------------------------------------------------------------
        if state.is_drawing_poly:
            if event == cv2.EVENT_LBUTTONDOWN:
                nx = round(x / w, 4)
                ny = round(y / h, 4)

                # Snap-to-close if clicked near first point and >= 3 pts
                if len(state.drawn_pts) >= 3:
                    fx = int(state.drawn_pts[0][0] * w)
                    fy = int(state.drawn_pts[0][1] * h)
                    if abs(x - fx) <= 18 and abs(y - fy) <= 18:
                        state.roi_pts = [list(p) for p in state.drawn_pts]
                        state.drawn_pts = []
                        state.is_drawing_poly = False
                        state.reset_motion()
                        state.last_side_result = None
                        state.show_toast(f"Custom ROI polygon applied ({len(state.roi_pts)} pts)! Press 'w' to save to file.")
                        print(f"[ROI] Closed custom polygon with {len(state.roi_pts)} vertices.")
                        return

                state.drawn_pts.append([nx, ny])
                print(f"[ROI] Vertex #{len(state.drawn_pts)} placed: ({nx:.4f}, {ny:.4f})")

            elif event == cv2.EVENT_RBUTTONDOWN:
                # Right-click undoes last placed point
                if state.drawn_pts:
                    removed = state.drawn_pts.pop()
                    print(f"[ROI] Removed vertex {removed}. Remaining: {len(state.drawn_pts)}")
            return

        # -------------------------------------------------------------------
        # Mode 2: Drag Rectangle ROI Mode ('d')
        # -------------------------------------------------------------------
        if state.is_dragging_roi:
            if event == cv2.EVENT_LBUTTONDOWN:
                state.roi_drag_start = (x, y)
                state.roi_drag_preview = None
            elif event == cv2.EVENT_MOUSEMOVE and state.roi_drag_start is not None:
                sx, sy = state.roi_drag_start
                state.roi_drag_preview = (min(sx, x), min(sy, y), max(sx, x), max(sy, y))
            elif event == cv2.EVENT_LBUTTONUP and state.roi_drag_start is not None:
                sx, sy = state.roi_drag_start
                x1, y1, x2, y2 = min(sx, x), min(sy, y), max(sx, x), max(sy, y)
                state.roi_drag_start = None
                state.roi_drag_preview = None
                if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                    nx1, ny1 = round(x1 / w, 4), round(y1 / h, 4)
                    nx2, ny2 = round(x2 / w, 4), round(y2 / h, 4)
                    state.roi_pts = [
                        [nx1, ny1],
                        [nx2, ny1],
                        [nx2, ny2],
                        [nx1, ny2]
                    ]
                    state.is_dragging_roi = False
                    state.reset_motion()
                    state.last_side_result = None
                    state.show_toast(f"Custom Box ROI applied! [{nx1},{ny1}]->[{nx2},{ny2}]. Press 'w' to save.")
                    print(f"[ROI] Applied custom box ROI: {state.roi_pts}")
                else:
                    print("[ROI] Drag box was too small; cancelled.")
            return

        # -------------------------------------------------------------------
        # Mode 3: Normal Mode -> Right-Click Drag for Manual Test Bounding Box
        # -------------------------------------------------------------------
        if event == cv2.EVENT_RBUTTONDOWN or (event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY)):
            state.test_box_drag_start = (x, y)
            state.manual_bbox = None
        elif event == cv2.EVENT_MOUSEMOVE and state.test_box_drag_start is not None:
            sx, sy = state.test_box_drag_start
            state.manual_bbox = (min(sx, x), min(sy, y), max(sx, x), max(sy, y))
        elif event in (cv2.EVENT_RBUTTONUP, cv2.EVENT_LBUTTONUP) and state.test_box_drag_start is not None:
            sx, sy = state.test_box_drag_start
            x1, y1, x2, y2 = min(sx, x), min(sy, y), max(sx, x), max(sy, y)
            state.test_box_drag_start = None
            if (x2 - x1) > 8 and (y2 - y1) > 8:
                state.manual_bbox = (x1, y1, x2, y2)
                res = classify_bbox_side(x1, y1, x2, y2, state.roi_pts, w, h)
                state.last_side_result = res
                print(f"[TEST BOX] ({x1},{y1})->({x2},{y2}) => {res['dumping_side'].upper()} "
                      f"(L: {res['left_px']:,} px | R: {res['right_px']:,} px | is_left: {res['is_left_side']})")
            else:
                state.manual_bbox = None

    return on_mouse


# ---------------------------------------------------------------------------
# Visual HUD Drawing
# ---------------------------------------------------------------------------
def draw_hud_and_overlays(frame: np.ndarray, state: TestState, active_res: dict | None) -> np.ndarray:
    h, w = frame.shape[:2]
    hud_h = 76
    strip = np.zeros((hud_h, w, 3), dtype=np.uint8)

    # Top separator line for HUD
    cv2.line(strip, (0, 0), (w, 0), (80, 80, 80), 1)

    now = time.time()
    res = active_res or state.last_side_result

    # 1. Mode Status / Toast Notification
    if now < state.toast_until and state.toast_msg:
        cv2.putText(strip, f"★ {state.toast_msg}", (16, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, state.toast_col, 2, cv2.LINE_AA)
        cv2.putText(strip, "Press [w] to write to disk  |  [r] redraw  |  [d] drag box", (16, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)
    elif state.is_drawing_poly:
        cv2.putText(strip, "CUSTOM POLYGON DRAWING ACTIVE", (16, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.68, (0, 255, 255), 2, cv2.LINE_AA)
        pts_cnt = len(state.drawn_pts)
        cv2.putText(strip, f"Points: {pts_cnt}  (Left-click: add | Right-click: undo | Click P1 or press [r]/Enter: finish | [c]: clear)",
                    (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 230), 1, cv2.LINE_AA)
    elif state.is_dragging_roi:
        cv2.putText(strip, "CUSTOM RECTANGLE ROI DRAG ACTIVE", (16, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.68, (50, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(strip, "Click and drag anywhere on screen to draw new ROI rectangle. Release to apply. [d] cancel",
                    (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 230), 1, cv2.LINE_AA)
    elif res and res.get("dumping_side") not in ("none", "unknown"):
        side = res.get("dumping_side", "unknown").upper()
        is_left = res.get("is_left_side", False)
        l_px = res.get("left_px", 0)
        r_px = res.get("right_px", 0)
        bx = res.get("bisector_x", 0)

        tag_col = (0, 210, 255) if side == "LEFT" else (50, 140, 255)  # Cyan vs Orange
        cv2.putText(strip, f"DUMPING SIDE: {side}", (16, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, tag_col, 2, cv2.LINE_AA)

        detail_text = (f"is_left_side: {is_left}   |   Left: {l_px:,} px   |   "
                       f"Right: {r_px:,} px   |   Bisector X: {bx}px")
        cv2.putText(strip, detail_text, (16, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
    elif res and res.get("dumping_side") == "none":
        cv2.putText(strip, "DETECTION OUTSIDE ROI", (16, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.68, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(strip, "Zero coverage in ROI (no side tag awarded)", (16, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
    else:
        cv2.putText(strip, "ROI SIDE MONITOR ACTIVE", (16, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.68, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(strip, "Wave hand in ROI or Right-Click Drag to test a bounding box", (16, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1, cv2.LINE_AA)

    # Status indicators on the right of the strip
    mot_col = (0, 230, 100) if state.motion_enabled else (90, 90, 90)
    mot_txt = "MOTION [b]: ON" if state.motion_enabled else "MOTION [b]: OFF"
    cv2.putText(strip, mot_txt, (w - 180, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, mot_col, 1, cv2.LINE_AA)
    cv2.putText(strip, "[r] Poly  [d] Box  [w] Save", (w - 230, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 170, 170), 1, cv2.LINE_AA)

    composed = np.vstack([frame, strip])

    # Drawing in progress preview (Polygon)
    if state.is_drawing_poly and state.drawn_pts:
        pts = [[int(p[0] * w), int(p[1] * h)] for p in state.drawn_pts]
        for idx, pt in enumerate(pts):
            pt_col = (0, 255, 0) if (idx == 0 and len(pts) >= 3) else (0, 180, 255)
            cv2.circle(composed, tuple(pt), 6, pt_col, -1)
            cv2.putText(composed, f"P{idx+1}", (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if len(pts) >= 2:
            cv2.polylines(composed, [np.array(pts, dtype=np.int32)], False, (0, 255, 255), 2)
            if len(pts) >= 3:
                cv2.line(composed, tuple(pts[-1]), tuple(pts[0]), (0, 200, 120), 1)

    # Drawing in progress preview (Rectangle drag)
    if state.is_dragging_roi and state.roi_drag_preview:
        rx1, ry1, rx2, ry2 = state.roi_drag_preview
        cv2.rectangle(composed, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
        cv2.putText(composed, f"New ROI Box: {rx2-rx1}x{ry2-ry1}", (rx1, max(24, ry1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)

    # Manual test bounding box
    if state.manual_bbox:
        x1, y1, x2, y2 = state.manual_bbox
        box_side = (state.last_side_result.get("dumping_side") if state.last_side_result else "test")
        box_col = (0, 210, 255) if box_side == "left" else ((50, 140, 255) if box_side == "right" else (255, 255, 0))
        cv2.rectangle(composed, (x1, y1), (x2, y2), box_col, 2)
        cv2.putText(composed, f"Test Box: {box_side.upper()}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_col, 1, cv2.LINE_AA)

    # Optional Help Box Overlay ('h')
    if state.show_help:
        help_lines = [
            "  ROI CONFIGURATION & TESTING:",
            "  r       - Toggle Polygon Draw Mode (Left-click points, Right-click undo)",
            "  d       - Toggle Rectangle Drag ROI (Click & drag new box directly)",
            "  w       - Save active custom ROI to roi_polygon.json",
            "  l       - Reload ROI from roi_polygon.json",
            "  1..4    - Load Presets: 1:Full, 2:Road Center, 3:Verge, 4:Right",
            "  c       - Clear in-progress polygon points",
            "  Enter   - Finish/confirm polygon points (when drawing)",
            "",
            "  DETECTION & CAPTURES:",
            "  b       - Toggle background subtraction motion detection",
            "  s       - Save snapshot (.jpg) + metadata sidecar (.json)",
            "  Right-Click Drag: Test any bounding box immediately",
            "  h       - Toggle this help window",
            "  q / ESC - Exit",
        ]
        box_h = len(help_lines) * 21 + 18
        box_w = 460
        overlay_box = np.full((box_h, box_w, 3), 20, dtype=np.uint8)
        cv2.rectangle(overlay_box, (0, 0), (box_w - 1, box_h - 1), (110, 110, 110), 1)
        for i, line in enumerate(help_lines):
            c = (0, 230, 255) if i in (0, 9) else (225, 225, 225)
            cv2.putText(overlay_box, line, (12, 19 + i * 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, c, 1, cv2.LINE_AA)
        tx = max(10, w - box_w - 16)
        composed[14 : 14 + box_h, tx : tx + box_w] = overlay_box

    return composed


# ---------------------------------------------------------------------------
# Static Synthetic Test Mode
# ---------------------------------------------------------------------------
def run_static_test(roi_override: list | None = None, save_report: bool = True):
    """Run automated geometric tests across multiple synthetic detection cases."""
    w, h = 640, 360
    roi = roi_override or parse_roi_input(None, None)

    pixel_pts = np.array([
        [int(p[0] * w) if p[0] <= 1.0 else int(p[0]),
         int(p[1] * h) if p[1] <= 1.0 else int(p[1])]
        for p in roi
    ], dtype=np.int32)
    rx, ry, rw, rh = cv2.boundingRect(pixel_pts)
    M = cv2.moments(pixel_pts)
    bx = int(M['m10'] / M['m00']) if M['m00'] != 0 else int(np.mean(pixel_pts[:, 0]))

    left_span = max(10, bx - rx)
    right_span = max(10, (rx + rw) - bx)

    test_cases = [
        {
            "name": "Pure Left Box",
            "bbox": [rx + 2, ry + 4, bx - 2, ry + rh - 4],
            "expected": "left"
        },
        {
            "name": "Pure Right Box",
            "bbox": [bx + 2, ry + 4, rx + rw - 2, ry + rh - 4],
            "expected": "right"
        },
        {
            "name": "Overlap Box (Majority Left)",
            "bbox": [rx + 2, ry + 4, bx + int(0.25 * right_span), ry + rh - 4],
            "expected": "left"
        },
        {
            "name": "Overlap Box (Majority Right)",
            "bbox": [bx - int(0.25 * left_span), ry + 4, rx + rw - 2, ry + rh - 4],
            "expected": "right"
        },
        {
            "name": "Outside ROI Box",
            "bbox": [10, 2, 80, min(30, max(5, ry - 4))] if ry > 15 else ([10, ry + rh + 4, 80, h - 5] if (h - (ry + rh)) > 15 else [0, 0, 5, 5]),
            "expected": "none"
        },
    ]

    print("\n" + "=" * 70)
    print("           ROI SIDE CLASSIFIER - STATIC GEOMETRIC TEST")
    print("=" * 70)
    print(f"Canvas Resolution : {w}x{h}")
    print(f"ROI Vertices      : {len(roi)} points -> {roi}")
    print(f"ROI Bounding Box  : ({rx}, {ry}, {rw}, {rh}) | Bisector X: {bx}px\n")

    canvas = np.full((h, w, 3), 25, dtype=np.uint8)

    all_passed = True
    results = []

    for idx, tc in enumerate(test_cases, 1):
        x1, y1, x2, y2 = tc["bbox"]
        res = classify_bbox_side(x1, y1, x2, y2, roi, w, h)
        side = res["dumping_side"]
        is_left = res["is_left_side"]
        l_px = res["left_px"]
        r_px = res["right_px"]
        bisector_x = res["bisector_x"]
        passed = (side == tc["expected"])
        if not passed:
            all_passed = False

        status_str = "PASS [OK]" if passed else "FAIL [X]"
        print(f"[{idx}] {tc['name']:<32}")
        print(f"    BBox        : ({x1}, {y1}) -> ({x2}, {y2})")
        print(f"    Bisector X  : {bisector_x} px")
        print(f"    Coverage    : Left = {l_px:>6} px | Right = {r_px:>6} px")
        print(f"    Result      : dumping_side={side!r:<7} is_left_side={is_left} | {status_str}")
        print("-" * 70)

        results.append({
            "test_case": tc["name"],
            "bbox": tc["bbox"],
            "result": res,
            "passed": passed
        })

        col = (0, 210, 255) if side == "left" else ((50, 140, 255) if side == "right" else (140, 140, 140))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
        cv2.putText(canvas, f"#{idx}:{side.upper()}", (x1, max(16, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

    draw_side_split_overlay(canvas, roi, alpha=0.30)

    summary_col = (0, 220, 80) if all_passed else (0, 0, 255)
    summary_text = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    cv2.putText(canvas, summary_text, (20, h - 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, summary_col, 2, cv2.LINE_AA)

    if save_report:
        os.makedirs(SAVE_DIR, exist_ok=True)
        img_out = os.path.join(SAVE_DIR, "static_test_result.jpg")
        json_out = os.path.join(SAVE_DIR, "static_test_result.json")
        cv2.imwrite(img_out, canvas)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump({
                "tested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "all_passed": all_passed,
                "roi_polygon": roi,
                "cases": results,
            }, f, indent=2)
        print(f"\n[REPORT] Saved visual result: {img_out}")
        print(f"[REPORT] Saved JSON result  : {json_out}\n")

    return all_passed, canvas


# ---------------------------------------------------------------------------
# Live Camera / Video Test Loop
# ---------------------------------------------------------------------------
def run_live(source: int | str, roi_initial: list, roi_file_path: str = DEFAULT_ROI_FILE):
    state = TestState(roi_initial, roi_path=roi_file_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video source: {source!r}")
        print("[TIP] You can test without a camera using: python test_roi_side_classifier.py --static")
        sys.exit(1)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or PROC_SIZE[0]
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or PROC_SIZE[1]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay = max(1, int(1000.0 / fps))
    is_file = isinstance(source, str) and not source.isdigit()

    os.makedirs(SAVE_DIR, exist_ok=True)
    cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_TITLE, 960, 600)
    cv2.setMouseCallback(WIN_TITLE, make_mouse_callback(state, cam_w, cam_h), {"w": cam_w, "h": cam_h})

    print(f"\n[START] Opened source: {source!r} ({cam_w}x{cam_h} @ {fps:.1f} fps)")
    print(f"[START] Initial ROI: {len(state.roi_pts)} points")
    print( "[START] Hotkeys for Custom ROI:")
    print("        [r] Draw custom polygon (Left-click add, Right-click undo)")
    print("        [d] Drag custom box ROI")
    print("        [w] Save active ROI to roi_polygon.json")
    print("        [1..4] Switch presets  |  [h] Toggle full help\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                state.reset_motion()
                continue
            print("[CAMERA] Stream ended.")
            break

        orig_h, orig_w = frame.shape[:2]

        # Downscale for motion processing
        pw, ph = PROC_SIZE
        proc = cv2.resize(frame, (pw, ph))
        sx, sy = orig_w / pw, orig_h / ph

        disp = frame.copy()

        # 1. Draw ROI side split overlay (idle state)
        if state.roi_pts and len(state.roi_pts) >= 3 and not (state.is_drawing_poly or state.is_dragging_roi):
            draw_side_split_overlay(disp, state.roi_pts, side_result=state.last_side_result, alpha=0.20)
            px_pts = [[int(p[0] * orig_w), int(p[1] * orig_h)] for p in state.roi_pts]
            cv2.polylines(disp, [np.array(px_pts, dtype=np.int32)], True, (0, 255, 200), 2)

        # 2. Motion Detection & Classification
        active_motion_result = None
        if state.motion_enabled and state.roi_pts and len(state.roi_pts) >= 3 and not (state.is_drawing_poly or state.is_dragging_roi):
            gray = cv2.GaussianBlur(cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY), (9, 9), 0)
            state.frame_count += 1

            if state.bg_model is None:
                state.bg_model = gray.astype(np.float32)
            else:
                cv2.accumulateWeighted(gray, state.bg_model, BG_ALPHA)

            if state.frame_count >= WARMUP_FRAMES:
                bg8 = cv2.convertScaleAbs(state.bg_model)
                diff = cv2.absdiff(bg8, gray)
                _, thr = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

                roi_mask = np.zeros((ph, pw), dtype=np.uint8)
                rmask_pts = [
                    [int(p[0] * pw) if p[0] <= 1.0 else int(p[0]),
                     int(p[1] * ph) if p[1] <= 1.0 else int(p[1])]
                    for p in state.roi_pts
                ]
                cv2.fillPoly(roi_mask, [np.array(rmask_pts, dtype=np.int32)], 255)
                thr = cv2.bitwise_and(thr, roi_mask)

                thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                dilated = cv2.dilate(thr, None, iterations=2)
                cnts = imutils.grab_contours(
                    cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                )

                side_votes = {"left": 0, "right": 0}
                motion_found = False
                last_sr = None

                for c in cnts:
                    if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                        continue
                    motion_found = True
                    x, y, w_box, h_box = cv2.boundingRect(c)
                    ox, oy = int(x * sx), int(y * sy)
                    ow_b, oh_b = int(w_box * sx), int(h_box * sy)

                    sr = classify_contour_side(c, state.roi_pts, pw, ph)
                    if sr["dumping_side"] in side_votes:
                        side_votes[sr["dumping_side"]] += (sr["left_px"] + sr["right_px"])
                    last_sr = sr

                    box_col = (0, 210, 255) if sr["dumping_side"] == "left" else (50, 140, 255)
                    cv2.rectangle(disp, (ox, oy), (ox + ow_b, oy + oh_b), box_col, 2)

                if motion_found and last_sr:
                    dom_side = "left" if side_votes["left"] >= side_votes["right"] else "right"
                    active_motion_result = {
                        "dumping_side": dom_side,
                        "is_left_side": (dom_side == "left"),
                        "left_px":      side_votes["left"],
                        "right_px":     side_votes["right"],
                        "bisector_x":   last_sr["bisector_x"],
                    }
                    state.last_side_result = active_motion_result
                    draw_side_split_overlay(disp, state.roi_pts, side_result=active_motion_result, alpha=0.32)

        # 3. Render HUD and composed frame
        composed = draw_hud_and_overlays(disp, state, active_motion_result)
        cv2.imshow(WIN_TITLE, composed)

        # 4. Handle Keypresses
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break

        # [r] Toggle Polygon Drawing Mode
        elif key == ord("r"):
            if not state.is_drawing_poly:
                state.is_drawing_poly = True
                state.is_dragging_roi = False
                state.drawn_pts = []
                state.reset_motion()
                print("[ROI] Polygon Drawing Mode ON. Left-click points, Right-click undo, click P1 to close.")
            else:
                if len(state.drawn_pts) >= 3:
                    state.roi_pts = [list(p) for p in state.drawn_pts]
                    state.drawn_pts = []
                    state.is_drawing_poly = False
                    state.reset_motion()
                    state.last_side_result = None
                    state.show_toast(f"Custom ROI polygon applied ({len(state.roi_pts)} pts)! Press 'w' to save.")
                    print(f"[ROI] Polygon saved with {len(state.roi_pts)} points.")
                else:
                    print(f"[ROI] Need >= 3 points to form polygon (have {len(state.drawn_pts)}).")

        # [Enter] or [Space]: confirm polygon while drawing
        elif key in (13, 32):  # Enter or Space
            if state.is_drawing_poly:
                if len(state.drawn_pts) >= 3:
                    state.roi_pts = [list(p) for p in state.drawn_pts]
                    state.drawn_pts = []
                    state.is_drawing_poly = False
                    state.reset_motion()
                    state.last_side_result = None
                    state.show_toast(f"Custom ROI applied ({len(state.roi_pts)} pts)! Press 'w' to save.")
                    print(f"[ROI] Custom ROI confirmed: {len(state.roi_pts)} vertices.")
                else:
                    print(f"[ROI] Need >= 3 points to confirm (currently {len(state.drawn_pts)}).")

        # [d] Toggle Drag Rectangle ROI Mode
        elif key == ord("d"):
            state.is_dragging_roi = not state.is_dragging_roi
            state.is_drawing_poly = False
            state.roi_drag_start = None
            state.roi_drag_preview = None
            if state.is_dragging_roi:
                print("[ROI] Drag ROI Mode ON. Click and drag on screen to set custom rectangle ROI.")
            else:
                print("[ROI] Drag ROI Mode OFF.")

        # [w] Write / Save Active ROI to JSON file
        elif key == ord("w"):
            if state.roi_pts and len(state.roi_pts) >= 3:
                saved_ok = save_roi_to_file(state.roi_pts, state.roi_path)
                if saved_ok:
                    state.show_toast(f"Saved custom ROI ({len(state.roi_pts)} pts) to {os.path.basename(state.roi_path)}!",
                                     col=(0, 255, 100))
                else:
                    state.show_toast("Failed to save ROI!", col=(0, 0, 255))
            else:
                state.show_toast("Cannot save: ROI has < 3 vertices!", col=(0, 0, 255))

        # [c] Clear in-progress points
        elif key == ord("c"):
            if state.is_drawing_poly:
                state.drawn_pts = []
                print("[ROI] Cleared in-progress points.")

        # [l] Reload ROI from disk
        elif key == ord("l"):
            state.roi_pts = parse_roi_input(state.roi_path, None, state.roi_path)
            state.reset_motion()
            state.last_side_result = None
            state.show_toast(f"Reloaded ROI from {os.path.basename(state.roi_path)} ({len(state.roi_pts)} pts)")

        # Quick Presets [1], [2], [3], [4]
        elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
            preset_key = chr(key)
            name, preset_pts = PRESET_ROIS[preset_key]
            state.roi_pts = [list(p) for p in preset_pts]
            state.is_drawing_poly = False
            state.is_dragging_roi = False
            state.reset_motion()
            state.last_side_result = None
            state.show_toast(f"Loaded Preset [{preset_key}]: {name}! Press 'w' to save.")
            print(f"[PRESET] Activated {name} ROI: {state.roi_pts}")

        # [b] Toggle Motion Detection
        elif key == ord("b"):
            state.motion_enabled = not state.motion_enabled
            state.reset_motion()
            state.show_toast(f"Motion Detection: {'ENABLED' if state.motion_enabled else 'DISABLED'}")
            print(f"[MOTION] Detection toggled to {'ON' if state.motion_enabled else 'OFF'}")

        # [h] Toggle Help
        elif key == ord("h"):
            state.show_help = not state.show_help

        # [s] Save Snapshot + Sidecar JSON
        elif key == ord("s"):
            state.snap_count += 1
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = os.path.join(SAVE_DIR, f"capture_{ts}_{state.snap_count:04d}")
            img_path = stem + ".jpg"
            json_path = stem + ".json"

            cv2.imwrite(img_path, disp)

            cur_sr = active_motion_result or state.last_side_result or {
                "dumping_side": "unknown",
                "is_left_side": False,
                "left_px": 0,
                "right_px": 0,
                "bisector_x": orig_w // 2,
            }
            sidecar_data = {
                "image_file":    os.path.basename(img_path),
                "captured_at":   datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "width":         orig_w,
                "height":        orig_h,
                "dumping_side":  cur_sr["dumping_side"],
                "is_left_side":  cur_sr["is_left_side"],
                "side_left_px":  cur_sr["left_px"],
                "side_right_px": cur_sr["right_px"],
                "roi_bisector_x": cur_sr["bisector_x"],
                "roi_polygon":   state.roi_pts,
                "snap_index":    state.snap_count,
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(sidecar_data, f, indent=2)

            state.show_toast(f"Snapshot #{state.snap_count} saved with side: {cur_sr['dumping_side'].upper()}!", col=(0, 220, 255))
            print(f"\n[SAVE] Saved capture #{state.snap_count}:")
            print(f"       Image : {img_path}")
            print(f"       JSON  : {json_path}")
            print(f"       TAG   : dumping_side={cur_sr['dumping_side']!r} | is_left_side={cur_sr['is_left_side']}\n")

    cap.release()
    cv2.destroyAllWindows()
    print("[EXIT] Test harness closed.")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Interactive and automated test harness for ROI dumping side classification with custom ROI support."
    )
    parser.add_argument("--source", "-s", default="0",
                        help="Camera index (0, 1...) or video file path. Default: 0")
    parser.add_argument("--roi", "-r", default=None,
                        help="Custom ROI JSON file path or raw JSON array string")
    parser.add_argument("--roi-box", default=None,
                        help="Custom rectangular ROI in normalized coordinates: x1,y1,x2,y2 (e.g. 0.1,0.2,0.9,0.8)")
    parser.add_argument("--static", action="store_true",
                        help="Run static automated tests (no camera required)")
    parser.add_argument("--headless", action="store_true",
                        help="Run static tests without opening an interactive GUI window")
    args = parser.parse_args()

    # Determine initial ROI
    roi_target_file = args.roi if (args.roi and os.path.isfile(args.roi)) else DEFAULT_ROI_FILE
    roi_initial = parse_roi_input(args.roi, args.roi_box, roi_target_file)

    if args.static or args.headless:
        passed, canvas = run_static_test(roi_override=roi_initial, save_report=True)
        if not args.headless:
            try:
                cv2.imshow("Static Test Results (Press any key to close)", canvas)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception:
                pass
        sys.exit(0 if passed else 1)

    src = int(args.source) if args.source.isdigit() else args.source
    run_live(src, roi_initial, roi_file_path=roi_target_file)


if __name__ == "__main__":
    main()
