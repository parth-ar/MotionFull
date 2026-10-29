from __future__ import annotations
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

# Standard preset ROIs for quick switching (normalised coordinates 0.0 - 1.0)
PRESET_ROIS = {
    "1": ("Full Frame", [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]]),
    "2": ("Road Center", [[0.20, 0.20], [0.80, 0.20], [0.80, 0.85], [0.20, 0.85]]),
    "3": ("Lower Verge", [[0.05, 0.40], [0.95, 0.40], [0.95, 0.95], [0.05, 0.95]]),
    "4": ("Right Verge", [[0.55, 0.25], [0.95, 0.25], [0.95, 0.85], [0.55, 0.85]]),
}


def _to_pixel_pts(roi_pts, frame_w: int, frame_h: int) -> np.ndarray:
    """Convert normalized (0-1) or absolute points to pixel coordinates."""
    pts = []
    for p in roi_pts:
        px = int(p[0] * frame_w) if p[0] <= 1.0 else int(p[0])
        py = int(p[1] * frame_h) if p[1] <= 1.0 else int(p[1])
        pts.append([px, py])
    return np.array(pts, dtype=np.int32)


def _polygon_centroid_x(pixel_pts: np.ndarray) -> int:
    """Compute the horizontal center of mass / bisector X of an ROI polygon."""
    M = cv2.moments(pixel_pts)
    if M['m00'] != 0:
        return int(M['m10'] / M['m00'])
    return int(np.mean(pixel_pts[:, 0]))


def _clip_polygon_to_half(pixel_pts: np.ndarray, bisector_x: int, side: str, frame_w: int, frame_h: int) -> np.ndarray:
    """Clip polygon to the left or right of the vertical bisector line."""
    if side == 'left':
        clip_rect = np.array([[0, 0], [bisector_x, 0],
                               [bisector_x, frame_h], [0, frame_h]], dtype=np.float32)
    else:
        clip_rect = np.array([[bisector_x, 0], [frame_w, 0],
                               [frame_w, frame_h], [bisector_x, frame_h]], dtype=np.float32)
    try:
        p1 = pixel_pts.astype(np.float32)
        retval, inter_pts = cv2.intersectConvexConvex(p1, clip_rect)
        if retval > 0 and inter_pts is not None and len(inter_pts) >= 3:
            return inter_pts.reshape(-1, 1, 2).astype(np.int32)
    except Exception:
        pass
    mask_roi  = np.zeros((frame_h, frame_w), dtype=np.uint8)
    mask_half = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillPoly(mask_roi,  [pixel_pts],                   255)
    cv2.fillPoly(mask_half, [clip_rect.astype(np.int32)],  255)
    inter = cv2.bitwise_and(mask_roi, mask_half)
    cnts, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        return max(cnts, key=cv2.contourArea)
    return np.array([], dtype=np.int32)


def _build_half_masks(roi_pts, frame_w: int, frame_h: int):
    """Build binary masks for the left and right halves of the ROI."""
    pixel_pts  = _to_pixel_pts(roi_pts, frame_w, frame_h)
    bisector_x = _polygon_centroid_x(pixel_pts)
    left_poly  = _clip_polygon_to_half(pixel_pts, bisector_x, 'left',  frame_w, frame_h)
    right_poly = _clip_polygon_to_half(pixel_pts, bisector_x, 'right', frame_w, frame_h)
    left_mask  = np.zeros((frame_h, frame_w), dtype=np.uint8)
    right_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    if left_poly is not None and len(left_poly) >= 3:
        cv2.fillPoly(left_mask,  [left_poly.reshape(-1, 2).astype(np.int32)],  255)
    if right_poly is not None and len(right_poly) >= 3:
        cv2.fillPoly(right_mask, [right_poly.reshape(-1, 2).astype(np.int32)], 255)
    return bisector_x, left_mask, right_mask


def _classify_from_masks(det_mask: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray, bisector_x: int) -> Dict[str, Any]:
    """Compare pixel overlap on left half vs right half."""
    left_px  = int(np.count_nonzero(cv2.bitwise_and(det_mask, left_mask)))
    right_px = int(np.count_nonzero(cv2.bitwise_and(det_mask, right_mask)))
    if left_px == 0 and right_px == 0:
        return {
            'dumping_side': 'none',
            'is_left_side': False,
            'left_px':      0,
            'right_px':     0,
            'bisector_x':   bisector_x,
        }
    is_left = left_px >= right_px
    return {
        'dumping_side': 'left' if is_left else 'right',
        'is_left_side': is_left,
        'left_px':      left_px,
        'right_px':     right_px,
        'bisector_x':   bisector_x,
    }


def classify_contour_side(contour: np.ndarray, roi_pts: list, frame_w: int, frame_h: int) -> Dict[str, Any]:
    """Classify a detected motion contour as occurring on the left or right side of the ROI."""
    if roi_pts is None or len(roi_pts) < 3:
        return {'dumping_side': 'unknown', 'is_left_side': False,
                'left_px': 0, 'right_px': 0, 'bisector_x': frame_w // 2}
    bisector_x, left_mask, right_mask = _build_half_masks(roi_pts, frame_w, frame_h)
    det_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.drawContours(det_mask, [contour], -1, 255, thickness=cv2.FILLED)
    return _classify_from_masks(det_mask, left_mask, right_mask, bisector_x)


def classify_bbox_side(x1: int, y1: int, x2: int, y2: int, roi_pts: list, frame_w: int, frame_h: int) -> Dict[str, Any]:
    """Classify a bounding box as occurring on the left or right side of the ROI."""
    if roi_pts is None or len(roi_pts) < 3:
        return {'dumping_side': 'unknown', 'is_left_side': False,
                'left_px': 0, 'right_px': 0, 'bisector_x': frame_w // 2}
    bisector_x, left_mask, right_mask = _build_half_masks(roi_pts, frame_w, frame_h)
    bx1 = max(0, min(frame_w, x1));  bx2 = max(0, min(frame_w, x2))
    by1 = max(0, min(frame_h, y1));  by2 = max(0, min(frame_h, y2))
    det_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    if bx2 > bx1 and by2 > by1:
        det_mask[by1:by2, bx1:bx2] = 255
    return _classify_from_masks(det_mask, left_mask, right_mask, bisector_x)


def draw_side_split_overlay(frame: np.ndarray, roi_pts: list, side_result: Optional[dict] = None,
                            alpha: float = 0.22, show_fill: bool = False) -> None:
    """
    Draws the device ROI split in the middle with a dotted line.
    
    By default, show_fill=False so NO translucent color fills obscure the camera feed,
    satisfying the requirement: 'i just want the device roi to be split in the middle,
    just a dotted line to indicate that'.
    """
    if roi_pts is None or len(roi_pts) < 3:
        return
    h, w = frame.shape[:2]
    pixel_pts  = _to_pixel_pts(roi_pts, w, h)
    bisector_x = _polygon_centroid_x(pixel_pts)

    # Optional color fill (disabled by default)
    if show_fill:
        left_poly  = _clip_polygon_to_half(pixel_pts, bisector_x, 'left',  w, h)
        right_poly = _clip_polygon_to_half(pixel_pts, bisector_x, 'right', w, h)
        overlay = frame.copy()
        dominant  = side_result.get('dumping_side', '') if side_result else ''
        left_col  = (0, 120, 255) if dominant == 'left'  else (0, 50, 100)
        right_col = (255, 80,  0) if dominant == 'right' else (80, 30,  0)
        if left_poly is not None and len(left_poly) >= 3:
            cv2.fillPoly(overlay, [left_poly.reshape(-1, 2).astype(np.int32)],  left_col)
        if right_poly is not None and len(right_poly) >= 3:
            cv2.fillPoly(overlay, [right_poly.reshape(-1, 2).astype(np.int32)], right_col)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    # Vertical dotted line splitting the ROI in the middle
    roi_ys   = pixel_pts[:, 1]
    top_y    = max(0, int(np.min(roi_ys)))
    bottom_y = min(h - 1, int(np.max(roi_ys)))
    dash_len = 6
    gap_len  = 6
    y = top_y
    while y < bottom_y:
        dash_end = min(y + dash_len, bottom_y)
        mid_pt_y = (y + dash_end) / 2.0
        # Only draw dashes that fall inside the ROI polygon
        if cv2.pointPolygonTest(pixel_pts, (float(bisector_x), mid_pt_y), False) >= 0:
            # Contrast outline (dark background for dash)
            cv2.line(frame, (bisector_x, y), (bisector_x, dash_end), (20, 20, 20), 2, cv2.LINE_AA)
            # Bright dotted line on top
            cv2.line(frame, (bisector_x, y), (bisector_x, dash_end), (230, 230, 230), 1, cv2.LINE_AA)
        y += dash_len + gap_len

    # Clean side indicators: 'L' and 'R' on either side of the bisector
    mid_y = (top_y + bottom_y) // 2
    dominant = (side_result.get('dumping_side', '') if side_result else '').lower()
    l_col = (0, 230, 255) if dominant == 'left' else (190, 190, 190)
    r_col = (50, 160, 255) if dominant == 'right' else (190, 190, 190)

    # Label L with drop shadow
    cv2.putText(frame, 'L', (max(0, bisector_x - 22), mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, 'L', (max(0, bisector_x - 22), mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, l_col, 1, cv2.LINE_AA)

    # Label R with drop shadow
    cv2.putText(frame, 'R', (min(w - 18, bisector_x + 8), mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, 'R', (min(w - 18, bisector_x + 8), mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, r_col, 1, cv2.LINE_AA)
