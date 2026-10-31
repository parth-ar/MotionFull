#!/usr/bin/env python3
"""
test_motion_thresholds.py — Real-Time Motion Threshold Tuning & ROI Adjustment Tool.

Designed specifically for interactive tuning on laptops (Windows & Linux) and edge devices.
Loads baseline configuration directly from config.py and roi_polygon.json, streams from
the camera, runs the exact production background-subtraction pipeline, and provides:

Features:
  1. Real-Time Sliders (Trackbars):
       - Diff Threshold       : Pixel difference sensitivity (DIFF_THRESHOLD)
       - Min Contour Area     : Minimum pixel area to trigger motion (MIN_CONTOUR_AREA)
       - BG Model Alpha %     : Learning rate for steady-state background (BG_ALPHA)
       - Fast BG Alpha %      : Learning rate during illumination shocks (FAST_BG_ALPHA)
       - Max Area Shock %     : Rejection ratio for global lighting shocks (MAX_MOTION_AREA_RATIO)
       - Consec Frames        : Consecutive frames required before trigger (MOTION_CONSECUTIVE_FRAMES)
       - Morph Kernel         : Morphological opening kernel size (noise/glint filtering)
       - Dilate Iters         : Dilation passes to bridge adjacent contour fragments

  2. Interactive ROI / Area of Interest Adjustment:
       - 'D' key : Draw Mode (Click points to draw custom polygon, Right-Click undo, Enter/Space finish)
       - 'E' key : Edit/Drag Mode (Click & drag existing vertices with mouse in real time)
       - '1'–'4' : Preset ROIs (Full Frame, Road Center, Lower Verge, Right Verge)
       - 'S' key : Save current ROI directly to roi_polygon.json (instantly usable by main app)
       - 'R' key : Reload/Reset ROI from roi_polygon.json or default

  3. Visualisation Modes ('V' key / Tab):
       - Mode 0 : Main View (Camera + ROI + Side Bisector + Motion Bounding Boxes + HUD)
       - Mode 1 : Binary Motion Mask (Pixels that passed diff, thresh, ROI, and morph)
       - Mode 2 : Grayscale Difference Image (|Frame - Background Model|)
       - Mode 3 : Quad View (2x2 Grid displaying Main, Diff, Thresh, and Dilated masks simultaneously!)

  4. Side Classification & Lighting Shock Rejection:
       - Real-time Left vs Right dumping side classification with center dotted bisector
       - Visual warning when global lighting shock / headlight glare triggers adaptation

  5. Laptop Optimised:
       - Auto-detects laptop webcam with DirectShow (cv2.CAP_DSHOW) on Windows for instant boot
       - Smooth 30 FPS rendering, responsive hotkeys, parameter export summary on exit

Usage:
    python test_motion_thresholds.py
    python test_motion_thresholds.py --source 0
    python test_motion_thresholds.py --source 1 --width 1280 --height 720
    python test_motion_thresholds.py --source test_video.mp4
"""

import argparse
import copy
import json
import math
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cv2
import imutils
import numpy as np

# Load baseline configurations directly from the main motion file
import config as _cfg
from motion import (
    load_roi_polygon, save_roi_polygon,
    apply_polygon_roi_mask,
    DEFAULT_ROI_POLYGON,
)
from roi_side_classifier import (
    classify_contour_side,
    draw_side_split_overlay,
    PRESET_ROIS,
)

WIN_MAIN = "SWSTP Motion Threshold & ROI Calibrator"
WIN_TRACKBARS = "Tuning Controls"


class MotionCalibrator:
    """Encapsulates real-time interactive motion threshold tuning and ROI editing."""

    def __init__(self, source=0, target_w=640, target_h=480, target_fps=30.0):
        self.source = source
        self.target_w = target_w
        self.target_h = target_h
        self.target_fps = target_fps

        # ── 1. Baseline parameters from config.py ────────────────────────────
        self.diff_threshold = int(getattr(_cfg, "DIFF_THRESHOLD", 30))
        self.min_contour_area = int(getattr(_cfg, "MIN_CONTOUR_AREA", 700))
        self.bg_alpha = float(getattr(_cfg, "BG_ALPHA", 0.05))
        self.fast_bg_alpha = float(getattr(_cfg, "FAST_BG_ALPHA", 0.25))
        self.max_motion_area_ratio = float(getattr(_cfg, "MAX_MOTION_AREA_RATIO", 0.55))
        self.consecutive_required = int(getattr(_cfg, "MOTION_CONSECUTIVE_FRAMES", 2))
        self.proc_size = getattr(_cfg, "FRAME_SIZE", (320, 180))
        self.morph_kernel_size = 3
        self.dilate_iters = 2

        # ── 2. ROI Polygon state ─────────────────────────────────────────────
        self.active_roi = copy.deepcopy(load_roi_polygon())
        self.roi_edit_mode = "NORMAL"   # "NORMAL", "DRAW", "DRAG"
        self.new_points = []            # points placed during DRAW mode
        self.mouse_pos = [0, 0]
        self.dragged_vertex_idx = None
        self.drag_hover_idx = None

        # ── 3. Runtime pipeline state ────────────────────────────────────────
        self.bg_model = None
        self.consecutive_motion_frames = 0
        self.motion_detected = False
        self.is_lighting_shock = False
        self.side_result = None
        self.candidate_contours = []
        self.largest_area = 0.0
        self.motion_pixels = 0
        self.roi_area = 1.0

        # Diagnostics & FPS
        self.view_mode = 0   # 0=Main, 1=Mask, 2=Diff, 3=Quad View
        self.show_help = True
        self.show_side_fill = False
        self.fps = 0.0
        self._fps_hist = []
        self._last_frame_time = time.monotonic()
        self.total_triggers = 0
        self.last_trigger_time = 0.0
        self.status_toast = "Ready. Adjust sliders or press 'H' for keybindings."
        self.status_toast_time = time.monotonic() + 4.0

        # Camera handle
        self.cap = None

    def open_camera(self) -> bool:
        """Open video source with laptop DirectShow acceleration on Windows."""
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        print(f"[*] Opening video capture on source: {src}...")

        if isinstance(src, int) and sys.platform.startswith("win"):
            # On Windows, DirectShow (CAP_DSHOW) opens laptop webcams immediately without 5s lag
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(src)
        else:
            self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            print(f"[!] ERROR: Failed to open video source '{src}'.")
            return False

        # Set properties
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_h)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Read test frame
        ok, frame = self.cap.read()
        if not ok or frame is None:
            print(f"[!] Warning: Opened '{src}' but could not grab test frame.")
            return False

        h, w = frame.shape[:2]
        print(f"[+] Camera online: {w}x{h} @ {self.cap.get(cv2.CAP_PROP_FPS):.1f} FPS")
        return True

    def init_gui(self):
        """Create OpenCV display window and real-time adjustment trackbars."""
        cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_MAIN, 1120, 630)
        cv2.setMouseCallback(WIN_MAIN, self._mouse_callback)

        # Trackbar Window
        cv2.namedWindow(WIN_TRACKBARS, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_TRACKBARS, 480, 360)

        # 1. Diff Threshold (5 - 100)
        cv2.createTrackbar("Diff Thresh", WIN_TRACKBARS, self.diff_threshold, 100, self._on_diff_trackbar)
        cv2.setTrackbarMin("Diff Thresh", WIN_TRACKBARS, 5)

        # 2. Min Contour Area (50 - 3000)
        cv2.createTrackbar("Min Area", WIN_TRACKBARS, self.min_contour_area, 3000, self._on_area_trackbar)
        cv2.setTrackbarMin("Min Area", WIN_TRACKBARS, 50)

        # 3. BG Alpha % (1 - 30 %)
        alpha_pct = max(1, min(30, int(round(self.bg_alpha * 100))))
        cv2.createTrackbar("BG Alpha %", WIN_TRACKBARS, alpha_pct, 30, self._on_alpha_trackbar)
        cv2.setTrackbarMin("BG Alpha %", WIN_TRACKBARS, 1)

        # 4. Fast BG Alpha % (10 - 60 %)
        fast_alpha_pct = max(10, min(60, int(round(self.fast_bg_alpha * 100))))
        cv2.createTrackbar("Fast Alpha %", WIN_TRACKBARS, fast_alpha_pct, 60, self._on_fast_alpha_trackbar)
        cv2.setTrackbarMin("Fast Alpha %", WIN_TRACKBARS, 10)

        # 5. Max Area Ratio % (10 - 95 %)
        max_ratio_pct = max(10, min(95, int(round(self.max_motion_area_ratio * 100))))
        cv2.createTrackbar("Max Area %", WIN_TRACKBARS, max_ratio_pct, 95, self._on_max_area_trackbar)
        cv2.setTrackbarMin("Max Area %", WIN_TRACKBARS, 10)

        # 6. Consecutive Frames (1 - 8)
        cv2.createTrackbar("Consec Frms", WIN_TRACKBARS, self.consecutive_required, 8, self._on_consec_trackbar)
        cv2.setTrackbarMin("Consec Frms", WIN_TRACKBARS, 1)

        # 7. Morph Open Kernel Size (1 - 7)
        cv2.createTrackbar("Morph Kernel", WIN_TRACKBARS, self.morph_kernel_size, 7, self._on_morph_trackbar)
        cv2.setTrackbarMin("Morph Kernel", WIN_TRACKBARS, 1)

        # 8. Dilate Iterations (1 - 5)
        cv2.createTrackbar("Dilate Iters", WIN_TRACKBARS, self.dilate_iters, 5, self._on_dilate_trackbar)
        cv2.setTrackbarMin("Dilate Iters", WIN_TRACKBARS, 1)

    # ── Trackbar Callbacks ───────────────────────────────────────────────────
    def _on_diff_trackbar(self, val):
        self.diff_threshold = max(5, val)

    def _on_area_trackbar(self, val):
        self.min_contour_area = max(50, val)

    def _on_alpha_trackbar(self, val):
        self.bg_alpha = max(0.01, val / 100.0)

    def _on_fast_alpha_trackbar(self, val):
        self.fast_bg_alpha = max(0.10, val / 100.0)

    def _on_max_area_trackbar(self, val):
        self.max_motion_area_ratio = max(0.10, val / 100.0)

    def _on_consec_trackbar(self, val):
        self.consecutive_required = max(1, val)

    def _on_morph_trackbar(self, val):
        # Force odd kernel sizes (1, 3, 5, 7)
        k = max(1, val)
        if k % 2 == 0:
            k += 1
        self.morph_kernel_size = k

    def _on_dilate_trackbar(self, val):
        self.dilate_iters = max(1, val)

    # ── Mouse Callback for ROI Editing ───────────────────────────────────────
    def _mouse_callback(self, event, x, y, flags, param):
        self.mouse_pos = [x, y]
        w = max(1, getattr(self, "disp_w", self.target_w))
        h = max(1, getattr(self, "disp_h", self.target_h))

        norm_x = min(1.0, max(0.0, x / float(w)))
        norm_y = min(1.0, max(0.0, y / float(h)))

        # 1. DRAW MODE
        if self.roi_edit_mode == "DRAW":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.new_points.append([norm_x, norm_y])
                self.set_toast(f"Point {len(self.new_points)} added. Right-Click to undo, Space to finish.")
            elif event == cv2.EVENT_RBUTTONDOWN:
                if self.new_points:
                    self.new_points.pop()
                    self.set_toast(f"Undid point. Remaining: {len(self.new_points)}")

        # 2. DRAG VERTEX MODE
        elif self.roi_edit_mode == "DRAG":
            # Check vertex proximity (hover threshold ~18 pixels)
            if event == cv2.EVENT_MOUSEMOVE and self.dragged_vertex_idx is None:
                hover_idx = None
                for i, p in enumerate(self.active_roi):
                    px = int(p[0] * w) if p[0] <= 1.0 else int(p[0])
                    py = int(p[1] * h) if p[1] <= 1.0 else int(p[1])
                    if math.hypot(x - px, y - py) <= 18:
                        hover_idx = i
                        break
                self.drag_hover_idx = hover_idx

            elif event == cv2.EVENT_LBUTTONDOWN:
                for i, p in enumerate(self.active_roi):
                    px = int(p[0] * w) if p[0] <= 1.0 else int(p[0])
                    py = int(p[1] * h) if p[1] <= 1.0 else int(p[1])
                    if math.hypot(x - px, y - py) <= 18:
                        self.dragged_vertex_idx = i
                        self.set_toast(f"Dragging vertex P{i+1}")
                        break

            elif event == cv2.EVENT_MOUSEMOVE and self.dragged_vertex_idx is not None:
                idx = self.dragged_vertex_idx
                if 0 <= idx < len(self.active_roi):
                    self.active_roi[idx] = [round(norm_x, 4), round(norm_y, 4)]

            elif event == cv2.EVENT_LBUTTONUP:
                if self.dragged_vertex_idx is not None:
                    self.set_toast(f"Moved vertex P{self.dragged_vertex_idx+1}. Press 'S' to persist to file.")
                    self.dragged_vertex_idx = None

    def set_toast(self, msg: str, duration: float = 3.0):
        self.status_toast = msg
        self.status_toast_time = time.monotonic() + duration

    # ── Core Motion Detection Pipeline (Identical to main.py) ────────────────
    def process_frame(self, frame: np.ndarray):
        orig_h, orig_w = frame.shape[:2]
        proc_w, proc_h = self.proc_size
        scale_x = orig_w / float(proc_w)
        scale_y = orig_h / float(proc_h)

        # Preprocessing: Downsample to FRAME_SIZE (320x180) and apply 9x9 Gaussian Blur
        proc_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (9, 9), 0)

        # Initialize background model
        if self.bg_model is None:
            self.bg_model = gray_blur.astype(np.float32)

        # Accumulate steady-state background
        cv2.accumulateWeighted(gray_blur, self.bg_model, self.bg_alpha)

        # Frame difference and thresholding
        bg_uint8 = cv2.convertScaleAbs(self.bg_model)
        diff = cv2.absdiff(bg_uint8, gray_blur)
        _, thresh = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

        # Apply polygon ROI mask
        thresh = apply_polygon_roi_mask(thresh, self.active_roi, proc_w, proc_h)

        # Morphological opening (noise, glint, and specular reflection rejection)
        k_size = self.morph_kernel_size
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

        # Calculate active ROI area for lighting change evaluation
        if self.active_roi and len(self.active_roi) >= 3:
            pts_roi = [
                [int(p[0] * proc_w) if p[0] <= 1.0 else int(p[0]),
                 int(p[1] * proc_h) if p[1] <= 1.0 else int(p[1])]
                for p in self.active_roi
            ]
            self.roi_area = max(100.0, float(cv2.contourArea(np.array(pts_roi, dtype=np.int32))))
        else:
            self.roi_area = float(proc_w * proc_h)

        # Lighting shock evaluation
        self.motion_pixels = cv2.countNonZero(opened)
        ratio = self.motion_pixels / self.roi_area
        self.is_lighting_shock = (ratio > self.max_motion_area_ratio)

        self.candidate_contours = []
        self.largest_area = 0.0
        self.side_result = None

        if self.is_lighting_shock:
            # Accelerated adaptation to illumination shock
            cv2.accumulateWeighted(gray_blur, self.bg_model, self.fast_bg_alpha)
            self.consecutive_motion_frames = 0
            self.motion_detected = False
            dilated = opened.copy()
        else:
            # Morphological dilation to bridge adjacent contours
            dilated = cv2.dilate(opened, None, iterations=self.dilate_iters)
            cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            candidate_motion = False
            _side_votes = {"left": 0, "right": 0}

            for c in cnts:
                area = cv2.contourArea(c)
                if area < self.min_contour_area:
                    continue

                candidate_motion = True
                if area > self.largest_area:
                    self.largest_area = area

                (bx, by, bw, bh) = cv2.boundingRect(c)
                orig_bx = int(bx * scale_x)
                orig_by = int(by * scale_y)
                orig_bw = int(bw * scale_x)
                orig_bh = int(bh * scale_y)

                # Classify dumping side (Left vs Right)
                side = "unknown"
                try:
                    sr = classify_contour_side(c, self.active_roi, proc_w, proc_h)
                    side = sr.get("dumping_side", "unknown")
                    _side_votes["left"] += sr.get("left_px", 0)
                    _side_votes["right"] += sr.get("right_px", 0)
                except Exception:
                    pass

                self.candidate_contours.append({
                    "bbox": (orig_bx, orig_by, orig_bw, orig_bh),
                    "area": int(area),
                    "side": side,
                })

            if candidate_motion:
                self.consecutive_motion_frames += 1
            else:
                self.consecutive_motion_frames = 0

            # Determine dominant side
            if _side_votes["left"] > 0 or _side_votes["right"] > 0:
                dom_side = "left" if _side_votes["left"] >= _side_votes["right"] else "right"
            else:
                dom_side = "unknown"

            self.side_result = {
                "dumping_side": dom_side,
                "is_left_side": (dom_side == "left"),
                "left_px": _side_votes["left"],
                "right_px": _side_votes["right"],
            }

            self.motion_detected = (self.consecutive_motion_frames >= self.consecutive_required)
            if self.motion_detected:
                now_m = time.monotonic()
                if now_m - self.last_trigger_time > 1.0:
                    self.total_triggers += 1
                    self.last_trigger_time = now_m

        return diff, thresh, opened, dilated

    # ── Drawing and HUD Overlays ─────────────────────────────────────────────
    def draw_overlays(self, frame: np.ndarray, diff_img, thresh_img, dilated_img) -> np.ndarray:
        h, w = frame.shape[:2]
        self.disp_w, self.disp_h = w, h
        canvas = frame.copy()

        # 1. Draw Area of Interest polygon and middle dotted bisector
        if self.active_roi and len(self.active_roi) >= 3 and self.roi_edit_mode != "DRAW":
            draw_side_split_overlay(canvas, self.active_roi, self.side_result,
                                    alpha=0.20, show_fill=self.show_side_fill)

            # Draw polygon perimeter line
            pts_px = np.array([
                [int(p[0] * w) if p[0] <= 1.0 else int(p[0]),
                 int(p[1] * h) if p[1] <= 1.0 else int(p[1])]
                for p in self.active_roi
            ], dtype=np.int32)

            poly_color = (0, 220, 255) if not self.motion_detected else (0, 255, 0)
            cv2.polylines(canvas, [pts_px], isClosed=True, color=poly_color, thickness=2, lineType=cv2.LINE_AA)

            # Draw vertices in Edit/Drag Mode
            if self.roi_edit_mode == "DRAG":
                for i, pt in enumerate(pts_px):
                    is_hovered = (i == self.drag_hover_idx)
                    is_active = (i == self.dragged_vertex_idx)
                    v_col = (0, 0, 255) if is_active else ((0, 255, 255) if is_hovered else (0, 200, 100))
                    rad = 9 if (is_hovered or is_active) else 6
                    cv2.circle(canvas, tuple(pt), rad, v_col, -1)
                    cv2.circle(canvas, tuple(pt), rad + 2, (0, 0, 0), 1)
                    cv2.putText(canvas, f"P{i+1}", (pt[0] + 8, pt[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Draw in-progress polygon in DRAW mode
        if self.roi_edit_mode == "DRAW":
            pts_px = [
                (int(p[0] * w), int(p[1] * h))
                for p in self.new_points
            ]
            if len(pts_px) >= 2:
                cv2.polylines(canvas, [np.array(pts_px, dtype=np.int32)],
                              isClosed=False, color=(0, 255, 255), thickness=2)
                # Preview closing line
                cv2.line(canvas, pts_px[-1], pts_px[0], (0, 160, 255), 1, cv2.LINE_AA)
            if pts_px:
                # Preview cursor guide line
                cv2.line(canvas, pts_px[-1], tuple(self.mouse_pos), (0, 255, 100), 1)

            for i, pt in enumerate(pts_px):
                cv2.circle(canvas, pt, 6, (0, 255, 0), -1)
                cv2.putText(canvas, str(i + 1), (pt[0] + 8, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # 3. Draw detected motion bounding boxes
        for item in self.candidate_contours:
            (bx, by, bw, bh) = item["bbox"]
            side = item["side"]
            area = item["area"]

            box_col = (0, 210, 255) if side == "left" else ((50, 140, 255) if side == "right" else (0, 255, 0))
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), box_col, 2)

            label = f"{side.upper()} [{area}px]"
            cv2.putText(canvas, label, (bx, max(16, by - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (bx, max(16, by - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_col, 1, cv2.LINE_AA)

        # 4. Top Status Header Strip
        cv2.rectangle(canvas, (0, 0), (w, 52), (18, 18, 22), -1)
        cv2.line(canvas, (0, 52), (w, 52), (55, 55, 65), 1)

        # Status text & badge
        if self.is_lighting_shock:
            status_text = "LIGHTING SHOCK (Adapting fast)"
            status_col = (0, 140, 255)  # Orange
        elif self.motion_detected:
            d_side = self.side_result.get("dumping_side", "").upper() if self.side_result else ""
            status_text = f"MOTION DETECTED [{d_side}]" if d_side in ("LEFT", "RIGHT") else "MOTION DETECTED"
            status_col = (0, 255, 0)
        elif self.consecutive_motion_frames > 0:
            status_text = f"CANDIDATE (Frame {self.consecutive_motion_frames}/{self.consecutive_required})"
            status_col = (0, 215, 255)  # Yellow
        else:
            status_text = "STEADY / NO MOTION"
            status_col = (160, 160, 160)

        cv2.putText(canvas, status_text, (14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, status_text, (14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, status_col, 2, cv2.LINE_AA)

        # Metrics on header
        ratio_pct = (self.motion_pixels / max(1.0, self.roi_area)) * 100.0
        metrics = (f"FPS: {self.fps:.1f} | Area: {int(self.largest_area)}px (min: {self.min_contour_area}) | "
                   f"Shock: {ratio_pct:.1f}% (max: {int(self.max_motion_area_ratio*100)}%) | "
                   f"Triggers: {self.total_triggers}")
        cv2.putText(canvas, metrics, (w - 560, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

        # 5. Bottom Parameter Strip
        cv2.rectangle(canvas, (0, h - 38), (w, h), (18, 18, 22), -1)
        cv2.line(canvas, (0, h - 38), (w, h - 38), (55, 55, 65), 1)

        params_summary = (
            f"Diff: {self.diff_threshold} | "
            f"MinArea: {self.min_contour_area}px | "
            f"Alpha: {self.bg_alpha:.2f} | "
            f"Shock: {self.max_motion_area_ratio:.2f} | "
            f"Frames: {self.consecutive_required} | "
            f"Kernel: {self.morph_kernel_size}x{self.morph_kernel_size} | "
            f"Dilate: {self.dilate_iters}x | "
            f"Mode: {self.roi_edit_mode}"
        )
        cv2.putText(canvas, params_summary, (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)

        # 6. Toast Notification
        if time.monotonic() < self.status_toast_time:
            tw = min(w - 40, int(len(self.status_toast) * 8.5) + 30)
            tx = (w - tw) // 2
            cv2.rectangle(canvas, (tx, 60), (tx + tw, 90), (30, 30, 30), -1)
            cv2.rectangle(canvas, (tx, 60), (tx + tw, 90), (0, 200, 255), 1)
            cv2.putText(canvas, self.status_toast, (tx + 12, 81),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 7. Help Overlay (Toggle with 'H')
        if self.show_help:
            help_lines = [
                "── HOTKEYS & CONTROLS ──",
                " [ / ]   : -1 / +1 Diff Threshold",
                " - / +   : -50 / +50 Min Contour Area",
                " B       : Reset Background Model",
                " D       : Draw New ROI (L-Click points, R-Click undo, Space finish)",
                " E       : Drag/Edit ROI Vertices (Click & drag points with mouse)",
                " 1–4     : Preset ROIs (1:Full, 2:Center, 3:Lower, 4:Right)",
                " S       : Save ROI to roi_polygon.json",
                " R       : Reload saved ROI from file",
                " V / Tab : Cycle View (Main -> Mask -> Diff -> Quad)",
                " F       : Toggle Side Fill Color",
                " P       : Print Config Snippet for config.py",
                " H       : Toggle this Help Menu",
                " Q / Esc : Exit",
            ]
            panel_w = 420
            panel_h = len(help_lines) * 20 + 20
            px = w - panel_w - 15
            py = 65
            sub = canvas[py:py + panel_h, px:px + panel_w]
            overlay_bg = np.full_like(sub, 20)
            cv2.addWeighted(overlay_bg, 0.85, sub, 0.15, 0, sub)
            cv2.rectangle(canvas, (px, py), (px + panel_w, py + panel_h), (80, 80, 95), 1)

            for idx, line in enumerate(help_lines):
                col = (0, 220, 255) if idx == 0 else (220, 220, 220)
                cv2.putText(canvas, line, (px + 12, py + 20 + idx * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

        # ── Output View Selection ────────────────────────────────────────────
        if self.view_mode == 0:
            return canvas
        elif self.view_mode == 1:
            # Binary Mask View
            mask_bgr = cv2.cvtColor(dilated_img, cv2.COLOR_GRAY2BGR)
            cv2.putText(mask_bgr, "VIEW: BINARY MOTION MASK (Dilated)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            return mask_bgr
        elif self.view_mode == 2:
            # Grayscale Difference View (Amplified for visual clarity)
            diff_amp = cv2.convertScaleAbs(diff_img, alpha=2.0)
            diff_bgr = cv2.cvtColor(diff_amp, cv2.COLOR_GRAY2BGR)
            cv2.putText(diff_bgr, f"VIEW: RAW ABSDIFF (Diff Thresh = {self.diff_threshold})", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
            return diff_bgr
        elif self.view_mode == 3:
            # Quad View (2x2 Grid)
            half_w = w // 2
            half_h = h // 2

            p1 = cv2.resize(canvas, (half_w, half_h))
            cv2.putText(p1, "1. Camera + Overlays", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            p2 = cv2.resize(cv2.cvtColor(cv2.convertScaleAbs(diff_img, alpha=2.0), cv2.COLOR_GRAY2BGR), (half_w, half_h))
            cv2.putText(p2, f"2. Raw Diff (Thresh: {self.diff_threshold})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

            p3 = cv2.resize(cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR), (half_w, half_h))
            cv2.putText(p3, "3. Thresholded ROI Mask", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            p4 = cv2.resize(cv2.cvtColor(dilated_img, cv2.COLOR_GRAY2BGR), (half_w, half_h))
            cv2.putText(p4, f"4. Morph Opened & Dilated (MinArea: {self.min_contour_area}px)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 150, 255), 1, cv2.LINE_AA)

            top_row = np.hstack([p1, p2])
            bot_row = np.hstack([p3, p4])
            quad = np.vstack([top_row, bot_row])
            return quad

        return canvas

    # ── Main Interactive Loop ────────────────────────────────────────────────
    def run(self):
        if not self.open_camera():
            print("[!] Could not initialize camera. Exiting.")
            return

        self.init_gui()
        print("\n" + "=" * 65)
        print("     SWSTP REAL-TIME MOTION THRESHOLD & ROI CALIBRATOR")
        print("=" * 65)
        print(" Controls:")
        print("   • Adjust sliders in 'Tuning Controls' window in real time")
        print("   • Press 'D' to draw a custom ROI polygon")
        print("   • Press 'E' to drag and adjust ROI vertices with your mouse")
        print("   • Press 'S' to save tuned ROI to roi_polygon.json")
        print("   • Press 'V' to cycle views (Main / Mask / Diff / Quad View)")
        print("   • Press 'P' to print config.py snippet")
        print("   • Press 'Q' or Esc to quit\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                # Compute FPS
                now = time.monotonic()
                dt = now - self._last_frame_time
                self._last_frame_time = now
                if dt > 0.001:
                    self._fps_hist.append(1.0 / dt)
                    if len(self._fps_hist) > 15:
                        self._fps_hist.pop(0)
                    self.fps = sum(self._fps_hist) / len(self._fps_hist)

                # Process motion
                diff, thresh, opened, dilated = self.process_frame(frame)

                # Render overlays
                display = self.draw_overlays(frame, diff, thresh, dilated)

                cv2.imshow(WIN_MAIN, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):  # Q or Esc
                    break

                # ── Key Handling ──
                elif key in (ord('['), ):  # Decrease Diff Thresh
                    self.diff_threshold = max(5, self.diff_threshold - 1)
                    cv2.setTrackbarPos("Diff Thresh", WIN_TRACKBARS, self.diff_threshold)
                    self.set_toast(f"DIFF_THRESHOLD = {self.diff_threshold}")

                elif key in (ord(']'), ):  # Increase Diff Thresh
                    self.diff_threshold = min(100, self.diff_threshold + 1)
                    cv2.setTrackbarPos("Diff Thresh", WIN_TRACKBARS, self.diff_threshold)
                    self.set_toast(f"DIFF_THRESHOLD = {self.diff_threshold}")

                elif key in (ord('-'), ord('_')):  # Decrease Min Area
                    self.min_contour_area = max(50, self.min_contour_area - 50)
                    cv2.setTrackbarPos("Min Area", WIN_TRACKBARS, self.min_contour_area)
                    self.set_toast(f"MIN_CONTOUR_AREA = {self.min_contour_area}")

                elif key in (ord('='), ord('+')):  # Increase Min Area
                    self.min_contour_area = min(3000, self.min_contour_area + 50)
                    cv2.setTrackbarPos("Min Area", WIN_TRACKBARS, self.min_contour_area)
                    self.set_toast(f"MIN_CONTOUR_AREA = {self.min_contour_area}")

                elif key in (ord('b'), ord('B')):  # Reset Background Model
                    self.bg_model = None
                    self.consecutive_motion_frames = 0
                    self.set_toast("Background model reset / re-adapting.")

                elif key in (ord('d'), ord('D')):  # Draw Mode
                    self.roi_edit_mode = "DRAW"
                    self.new_points = []
                    self.set_toast("DRAW MODE: Left-Click points for new ROI. Right-Click undo. Space to finish.")

                elif key in (ord('e'), ord('E')):  # Edit / Drag Mode
                    self.roi_edit_mode = "DRAG" if self.roi_edit_mode != "DRAG" else "NORMAL"
                    mode_name = "DRAG VERTEX MODE (Click & drag vertices with mouse)" if self.roi_edit_mode == "DRAG" else "NORMAL MODE"
                    self.set_toast(mode_name)

                elif key in (32, 13):  # Space or Enter: Finish drawing
                    if self.roi_edit_mode == "DRAW":
                        if len(self.new_points) >= 3:
                            self.active_roi = copy.deepcopy(self.new_points)
                            self.roi_edit_mode = "NORMAL"
                            self.set_toast(f"ROI updated with {len(self.active_roi)} points! Press 'S' to save.")
                        else:
                            self.set_toast("Need at least 3 points to create polygon!")

                elif key in (ord('c'), ord('C')):  # Clear points in Draw Mode
                    if self.roi_edit_mode == "DRAW":
                        self.new_points.clear()
                        self.set_toast("Cleared drawing points.")

                elif key in (ord('s'), ord('S')):  # Save ROI
                    save_roi_polygon(self.active_roi)
                    self.set_toast("ROI successfully saved to roi_polygon.json!")

                elif key in (ord('r'), ord('R')):  # Reload ROI
                    self.active_roi = copy.deepcopy(load_roi_polygon())
                    self.roi_edit_mode = "NORMAL"
                    self.set_toast("ROI reloaded from roi_polygon.json.")

                elif key in (ord('1'), ord('2'), ord('3'), ord('4')):  # Presets
                    preset_key = chr(key)
                    if preset_key in PRESET_ROIS:
                        name, pts = PRESET_ROIS[preset_key]
                        self.active_roi = copy.deepcopy(pts)
                        self.set_toast(f"Loaded Preset '{name}'")

                elif key in (ord('v'), ord('V'), 9):  # 'V' or Tab: Cycle View
                    self.view_mode = (self.view_mode + 1) % 4
                    modes = ["Main + Overlays", "Binary Motion Mask", "Raw Difference Image", "Quad View (2x2 Grid)"]
                    self.set_toast(f"View Mode: {modes[self.view_mode]}")

                elif key in (ord('f'), ord('F')):  # Toggle Side Fill
                    self.show_side_fill = not self.show_side_fill
                    self.set_toast(f"Side Color Fill: {'ON' if self.show_side_fill else 'OFF'}")

                elif key in (ord('h'), ord('H')):  # Toggle Help
                    self.show_help = not self.show_help

                elif key in (ord('p'), ord('P')):  # Print config snippet
                    self.print_config_snippet()

        except KeyboardInterrupt:
            pass
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            self.print_summary()

    def print_config_snippet(self):
        """Prints a clean Python configuration block suitable for config.py."""
        print("\n" + "=" * 65)
        print("          CALIBRATED CONFIGURATION BLOCK (for config.py)")
        print("=" * 65)
        print(f"DIFF_THRESHOLD            = {self.diff_threshold}")
        print(f"MIN_CONTOUR_AREA          = {self.min_contour_area}")
        print(f"BG_ALPHA                  = {self.bg_alpha:.3f}")
        print(f"FAST_BG_ALPHA             = {self.fast_bg_alpha:.3f}")
        print(f"MAX_MOTION_AREA_RATIO     = {self.max_motion_area_ratio:.2f}")
        print(f"MOTION_CONSECUTIVE_FRAMES = {self.consecutive_required}")
        print(f"MORPH_KERNEL_SIZE         = {self.morph_kernel_size}")
        print(f"DILATE_ITERATIONS         = {self.dilate_iters}")
        print("=" * 65 + "\n")
        self.set_toast("Calibrated config snippet printed to console!")

    def print_summary(self):
        """Prints a clean summary on exit comparing calibrated values to defaults."""
        orig_diff = getattr(_cfg, "DIFF_THRESHOLD", 30)
        orig_area = getattr(_cfg, "MIN_CONTOUR_AREA", 700)
        orig_alpha = getattr(_cfg, "BG_ALPHA", 0.05)
        orig_ratio = getattr(_cfg, "MAX_MOTION_AREA_RATIO", 0.55)
        orig_consec = getattr(_cfg, "MOTION_CONSECUTIVE_FRAMES", 2)

        print("\n" + "=" * 65)
        print("       MOTION THRESHOLD CALIBRATION SUMMARY REPORT")
        print("=" * 65)
        print(f" {'Parameter':<26} | {'Baseline':<12} | {'Calibrated':<12}")
        print("-" * 65)
        print(f" {'DIFF_THRESHOLD':<26} | {orig_diff:<12} | {self.diff_threshold:<12}")
        print(f" {'MIN_CONTOUR_AREA':<26} | {orig_area:<12} | {self.min_contour_area:<12}")
        print(f" {'BG_ALPHA':<26} | {orig_alpha:<12.3f} | {self.bg_alpha:<12.3f}")
        print(f" {'FAST_BG_ALPHA':<26} | {getattr(_cfg, 'FAST_BG_ALPHA', 0.25):<12.3f} | {self.fast_bg_alpha:<12.3f}")
        print(f" {'MAX_MOTION_AREA_RATIO':<26} | {orig_ratio:<12.2f} | {self.max_motion_area_ratio:<12.2f}")
        print(f" {'CONSECUTIVE_FRAMES':<26} | {orig_consec:<12} | {self.consecutive_required:<12}")
        print(f" {'MORPH_KERNEL_SIZE':<26} | {'3':<12} | {self.morph_kernel_size:<12}")
        print(f" {'DILATE_ITERATIONS':<26} | {'2':<12} | {self.dilate_iters:<12}")
        print("-" * 65)
        print(f" Total Confirmed Triggers: {self.total_triggers}")
        print(f" Active ROI Vertices     : {len(self.active_roi)} points")
        print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Motion Threshold Calibration & ROI Adjustment Tool for Laptops and Edge Devices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", "-s", default=0,
                        help="Video source: camera index (0, 1), video file, or stream URL (default: 0)")
    parser.add_argument("--width", type=int, default=640,
                        help="Target capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Target capture height (default: 480)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Target capture FPS (default: 30.0)")

    args = parser.parse_args()

    calibrator = MotionCalibrator(
        source=args.source,
        target_w=args.width,
        target_h=args.height,
        target_fps=args.fps,
    )
    calibrator.run()


if __name__ == "__main__":
    main()
