"""
test_motion_robustness.py — Comprehensive unit test for motion detection robustness.

Tests:
  1. Low-contrast shadow rejection (delta 20 < DIFF_THRESHOLD 30) -> No motion.
  2. Small reflection glints (< MIN_CONTOUR_AREA 700) -> Filtered out by opening & area threshold.
  3. Single-frame flash / transient reflection -> Rejected by MOTION_CONSECUTIVE_FRAMES (2).
  4. Sudden global lighting shock (> 55% ROI area) -> Rejected by MAX_MOTION_AREA_RATIO, triggers fast adaptation.
  5. Authentic physical dumping motion (localized, delta 80, area 1500 px, 3 frames) -> Successfully detected!
"""

import cv2
import numpy as np
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config as _cfg
from motion import apply_polygon_roi_mask

def run_pipeline_step(frame, bg_model, consecutive_motion_frames, roi_pts=None):
    proc_w, proc_h = _cfg.FRAME_SIZE
    proc_frame = cv2.resize(frame, (proc_w, proc_h))
    gray_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray_frame, (9, 9), 0)

    if bg_model is None:
        bg_model = gray_blur.astype(np.float32)
        return False, bg_model, 0, False

    cv2.accumulateWeighted(gray_blur, bg_model, _cfg.BG_ALPHA)

    bg_uint8 = cv2.convertScaleAbs(bg_model)
    diff = cv2.absdiff(bg_uint8, gray_blur)
    _, thresh = cv2.threshold(diff, _cfg.DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    thresh = apply_polygon_roi_mask(thresh, roi_pts, proc_w, proc_h)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

    if roi_pts and len(roi_pts) >= 3:
        pts_roi = [
            [int(p[0] * proc_w) if p[0] <= 1.0 else int(p[0]),
             int(p[1] * proc_h) if p[1] <= 1.0 else int(p[1])]
            for p in roi_pts
        ]
        roi_area = max(100.0, float(cv2.contourArea(np.array(pts_roi, dtype=np.int32))))
    else:
        roi_area = float(proc_w * proc_h)

    motion_pixels = cv2.countNonZero(opened)
    is_lighting_shock = (motion_pixels / roi_area) > _cfg.MAX_MOTION_AREA_RATIO

    if is_lighting_shock:
        cv2.accumulateWeighted(gray_blur, bg_model, _cfg.FAST_BG_ALPHA)
        consecutive_motion_frames = 0
        motion_detected = False
    else:
        dilated = cv2.dilate(opened, None, iterations=2)
        cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_motion = False
        for c in cnts:
            if cv2.contourArea(c) < _cfg.MIN_CONTOUR_AREA:
                continue
            candidate_motion = True

        if candidate_motion:
            consecutive_motion_frames += 1
        else:
            consecutive_motion_frames = 0

        motion_detected = (consecutive_motion_frames >= _cfg.MOTION_CONSECUTIVE_FRAMES)

    return motion_detected, bg_model, consecutive_motion_frames, is_lighting_shock


def main():
    print("=================================================================")
    print("  Testing Robust Motion Detection Filters")
    print("=================================================================")
    print(f"DIFF_THRESHOLD:            {_cfg.DIFF_THRESHOLD}")
    print(f"MIN_CONTOUR_AREA:          {_cfg.MIN_CONTOUR_AREA}")
    print(f"MAX_MOTION_AREA_RATIO:     {_cfg.MAX_MOTION_AREA_RATIO}")
    print(f"BG_ALPHA / FAST_BG_ALPHA:  {_cfg.BG_ALPHA} / {_cfg.FAST_BG_ALPHA}")
    print(f"MOTION_CONSECUTIVE_FRAMES: {_cfg.MOTION_CONSECUTIVE_FRAMES}")
    print("-----------------------------------------------------------------")

    w, h = 640, 360
    base_gray = 120
    base_frame = np.full((h, w, 3), base_gray, dtype=np.uint8)

    # Initialize background model
    bg_model = None
    consecutive = 0
    for _ in range(15):
        _, bg_model, consecutive, _ = run_pipeline_step(base_frame, bg_model, consecutive)

    assert bg_model is not None, "Background model should be initialized"
    print("[INIT] Background model converged successfully.")

    # -----------------------------------------------------------------------
    # TEST 1: Low-contrast shadow (delta 20 < DIFF_THRESHOLD 30)
    # -----------------------------------------------------------------------
    shadow_frame = base_frame.copy()
    # Shadow darkens a 100x100 patch by 20 grayscale points
    shadow_frame[100:200, 200:300] = base_gray - 20
    detected, bg_model, consecutive, shock = run_pipeline_step(shadow_frame, bg_model, consecutive)
    assert not detected and consecutive == 0, f"Shadow caused false positive! detected={detected}"
    print("[PASS] Test 1: Low-contrast shadow correctly rejected (no detection).")

    # -----------------------------------------------------------------------
    # TEST 2: Small reflection glints (small isolated bright spots)
    # -----------------------------------------------------------------------
    glint_frame = base_frame.copy()
    # Few small 4x4 reflection glints (each 16 pixels, well below 700 px area)
    glint_frame[50:54, 50:54] = 255
    glint_frame[70:74, 150:154] = 255
    detected, bg_model, consecutive, shock = run_pipeline_step(glint_frame, bg_model, consecutive)
    assert not detected and consecutive == 0, f"Small glints caused false positive! detected={detected}"
    print("[PASS] Test 2: Small specular glints correctly rejected.")

    # -----------------------------------------------------------------------
    # TEST 3: Single-frame transient flash (1 frame only)
    # -----------------------------------------------------------------------
    flash_frame = base_frame.copy()
    # Moderate patch (e.g. 80x80 px) with delta 60 for 1 frame only
    flash_frame[100:180, 200:280] = base_gray + 60
    detected, bg_model, consecutive, shock = run_pipeline_step(flash_frame, bg_model, consecutive)
    # Consecutive should be 1, but detected should still be False because MOTION_CONSECUTIVE_FRAMES=2
    assert not detected and consecutive == 1, f"1-frame flash triggered capture! detected={detected}"
    # Next frame back to normal -> resets consecutive to 0
    detected, bg_model, consecutive, shock = run_pipeline_step(base_frame, bg_model, consecutive)
    assert not detected and consecutive == 0, "Counter should reset when flash ends"
    print("[PASS] Test 3: 1-frame transient flash correctly blocked by persistence filter.")

    # -----------------------------------------------------------------------
    # TEST 4: Sudden global lighting shock (> 55% ROI area)
    # -----------------------------------------------------------------------
    headlight_frame = base_frame.copy()
    # Flashlights or headlights light up 80% of the frame
    headlight_frame[:] = 230
    detected, bg_model, consecutive, shock = run_pipeline_step(headlight_frame, bg_model, consecutive)
    assert shock is True, "Global lighting shock should be detected!"
    assert detected is False, "Lighting shock must not trigger motion detection!"
    assert consecutive == 0, "Consecutive motion frames must reset to 0 during shock!"
    print("[PASS] Test 4: Global lighting shock correctly identified and ignored (is_lighting_shock=True).")

    # -----------------------------------------------------------------------
    # TEST 5: Authentic localized motion (person/garbage bag moving for 3 frames)
    # -----------------------------------------------------------------------
    # Re-settle background to base
    bg_model = None
    consecutive = 0
    for _ in range(15):
        _, bg_model, consecutive, _ = run_pipeline_step(base_frame, bg_model, consecutive)

    # Person/trash moving: 100x120 patch, high delta (+80)
    real_motion_frame = base_frame.copy()
    real_motion_frame[80:200, 200:320] = base_gray + 80

    # Frame 1: candidate detected, consecutive = 1, detected = False
    det1, bg_model, consecutive, _ = run_pipeline_step(real_motion_frame, bg_model, consecutive)
    assert det1 is False and consecutive == 1, f"Frame 1 should be candidate: det={det1}, count={consecutive}"

    # Frame 2: sustained motion, consecutive = 2, detected = True!
    det2, bg_model, consecutive, _ = run_pipeline_step(real_motion_frame, bg_model, consecutive)
    assert det2 is True and consecutive == 2, f"Frame 2 should confirm motion: det={det2}, count={consecutive}"

    # Frame 3: continues motion
    det3, bg_model, consecutive, _ = run_pipeline_step(real_motion_frame, bg_model, consecutive)
    assert det3 is True and consecutive == 3, f"Frame 3 should continue motion: det={det3}, count={consecutive}"

    print("[PASS] Test 5: Authentic localized motion accurately detected on sustained movement!")
    print("-----------------------------------------------------------------")
    print("ALL 5 MOTION ROBUSTNESS TESTS PASSED SUCCESSFULLY!")
    print("=================================================================")

if __name__ == "__main__":
    main()
