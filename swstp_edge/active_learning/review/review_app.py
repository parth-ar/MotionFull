"""
Active-Learning Review App  (single-class edition)
---------------------------------------------------
Walks through a folder of images, shows the model's predicted boxes, and
lets you correct them with the mouse/keyboard. Confirmed frames are saved
in YOLO format (images/ + labels/) ready to merge into training data.

Two ways to feed it predictions:
  1. Point --weights at a .pt or .onnx file -> it runs inference itself.
  2. Point --preds-dir at a folder of YOLO-format .txt files -> reuses those.

Progress is saved to review_state.json so you can quit and resume later.

CONTROLS (shown on-screen):
  Left-drag or 2-click       Draw a new litter box (Shift+drag over existing boxes)
  Left-drag inside a box     Move the box
  Left-drag near a corner    Resize the box
  Right-click a box          Delete that box (or cancel in-progress drawing)
  Click a box to select it,
  then d / Del / Backspace   Delete the selected box
  c                          Clear ALL boxes (mark frame as clean / FP-only)
  s / Enter                  Accept -> save image + label, move to next
  n / Space                  Skip (not saved), move to next
  z                          Undo last added box
  u                          Reset frame to original model predictions
  q / Esc                    Quit (progress is saved)

Usage:
    python review_app.py --images-dir ../dataset/raw --weights ../../weights/best.pt \\
        --output ../dataset/reviewed/batch_01
"""

import argparse
import datetime
import json
import os
import sys

import cv2
import numpy as np

# Ensure UTF-8 output on Windows consoles with Unicode usernames
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure swstp_edge root is accessible so detector.py can be imported
_HERE = os.path.dirname(os.path.abspath(__file__))
_ACTIVE_LEARNING_DIR = os.path.abspath(os.path.join(_HERE, ".."))
_SWSTP_EDGE_DIR = os.path.abspath(os.path.join(_ACTIVE_LEARNING_DIR, ".."))
for p in (_SWSTP_EDGE_DIR, _ACTIVE_LEARNING_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

CORNER_GRAB_PX = 10
BOX_HIT_MARGIN  = 4
BOX_COLOR       = (0, 0, 220)       # thick red — litter detection boxes (BGR)
BOX_THICKNESS   = 3                 # normal box line thickness
SEL_COLOR       = (0, 255, 255)     # bright yellow — selected / drawing box highlight (BGR)
SEL_THICKNESS   = 3                 # selected box line thickness
CORNER_COLOR    = (0, 255, 255)     # bright yellow — resize grab dots
HUD_HEIGHT      = 44               # pixels reserved at top for the HUD bar
MAX_DISPLAY_W   = 1280             # max window width before downscaling for display
MAX_DISPLAY_H   = 800              # max window height before downscaling for display

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Unicode-safe imread / imwrite  (cv2.imread fails on non-ASCII Windows paths)
# ---------------------------------------------------------------------------

def imread_unicode(path):
    """
    Unicode-safe replacement for cv2.imread().
    cv2's C++ backend uses the system ANSI codepage on Windows and silently
    fails on paths that contain non-ASCII characters (e.g. Unicode usernames).
    Using Python's built-in open() avoids that limitation entirely.
    """
    try:
        with open(path, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, img):
    """
    Unicode-safe replacement for cv2.imwrite().
    Encodes in memory then writes bytes with Python's open().
    Returns True on success, False on failure.
    """
    ext = os.path.splitext(path)[1].lower() or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False


def fit_frame(frame):
    """
    Downscale `frame` for display if it exceeds MAX_DISPLAY dimensions.
    Returns (display_frame, scale_x, scale_y) where scales <= 1.0.

    CRITICAL — coordinate mapping:
    With WINDOW_AUTOSIZE, OpenCV delivers mouse coordinates in the coordinate
    space of the image passed to imshow(). Dividing incoming mouse coordinates
    by (scale_x, scale_y) accurately converts them back to the original image space.
    """
    h, w = frame.shape[:2]
    scale = min(MAX_DISPLAY_W / max(w, 1), MAX_DISPLAY_H / max(h, 1), 1.0)
    if scale < 1.0:
        dw = max(1, int(round(w * scale)))
        dh = max(1, int(round(h * scale)))
        scaled = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
        return scaled, dw / w, dh / h
    return frame, 1.0, 1.0


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_default_paths():
    """Resolve standard directory locations across swstp_edge layout."""
    today_str = datetime.date.today().strftime("%Y_%m_%d")

    raw_dataset     = os.path.join(_ACTIVE_LEARNING_DIR, "dataset", "raw")
    litter_captures = os.path.join(_SWSTP_EDGE_DIR, "captures", "litter")
    test_imgs       = os.path.join(_SWSTP_EDGE_DIR, "test images")

    images_dir = raw_dataset
    if not os.path.isdir(images_dir) or not any(
        f.lower().endswith(tuple(IMG_EXTS)) for f in os.listdir(images_dir)
    ):
        if os.path.isdir(litter_captures) and any(
            f.lower().endswith(tuple(IMG_EXTS)) for f in os.listdir(litter_captures)
        ):
            images_dir = litter_captures
        elif os.path.isdir(test_imgs):
            images_dir = test_imgs

    weights_pt = os.path.join(_SWSTP_EDGE_DIR, "weights", "best.pt")
    output_dir = os.path.join(_ACTIVE_LEARNING_DIR, "dataset", "reviewed", f"batch_{today_str}")
    return images_dir, weights_pt, output_dir


def parse_args():
    default_images, default_weights, default_output = resolve_default_paths()
    p = argparse.ArgumentParser(description="Active-learning single-class review app (litter)")
    p.add_argument("--images-dir", default=default_images,
                   help="Folder of images to review (default: active_learning/dataset/raw)")
    p.add_argument("--weights", default=default_weights,
                   help="YOLO weights (.pt or .onnx) for inference (default: swstp_edge/weights/best.pt)")
    p.add_argument("--preds-dir", default=None,
                   help="Pre-computed YOLO .txt predictions folder (optional, skips inference)")
    p.add_argument("--output", default=default_output,
                   help="Where to write reviewed images/ + labels/ (default: dataset/reviewed/batch_YYYY_MM_DD)")
    p.add_argument("--conf", type=float, default=0.15,
                   help="Min confidence threshold for live inference (default: 0.15)")
    p.add_argument("--classes", default="litter",
                   help=argparse.SUPPRESS)
    return p.parse_args()


# ---------------------------------------------------------------------------
# YOLO .txt I/O
# ---------------------------------------------------------------------------

def load_yolo_txt(path, img_w, img_h):
    """Read a YOLO-format .txt (5 or 6 columns) into pixel-space boxes."""
    boxes = []
    if not os.path.isfile(path):
        return boxes
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else None
            x1 = int((xc - bw / 2) * img_w)
            y1 = int((yc - bh / 2) * img_h)
            x2 = int((xc + bw / 2) * img_w)
            y2 = int((yc + bh / 2) * img_h)
            boxes.append({"cls": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf})
    return boxes


def save_yolo_txt(path, boxes, img_w, img_h):
    """Save bounding boxes to YOLO-format text file."""
    lines = []
    for b in boxes:
        x1, y1 = min(b["x1"], b["x2"]), min(b["y1"], b["y2"])
        x2, y2 = max(b["x1"], b["x2"]), max(b["y1"], b["y2"])
        if x2 <= x1 or y2 <= y1:
            continue
        xc = ((x1 + x2) / 2) / img_w
        yc = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        lines.append(f"{b['cls']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Review Session
# ---------------------------------------------------------------------------

class ReviewSession:
    """Manages the current frame's box state and all mouse interactions."""

    def __init__(self):
        self.boxes          = []
        self.orig_boxes     = []
        self.selected_idx   = None
        self.drag_mode      = None   # 'new_drag' | 'new_click' | 'move' | 'resize'
        self.drag_corner    = None
        self.drag_start     = None
        self.drag_box_start = None
        self.img_w          = 0
        self.img_h          = 0
        self.scale_x        = 1.0
        self.scale_y        = 1.0

    def load(self, boxes, img_w, img_h):
        self.boxes          = [dict(b) for b in boxes]
        self.orig_boxes     = [dict(b) for b in boxes]
        self.img_w          = max(1, img_w)
        self.img_h          = max(1, img_h)
        self.selected_idx   = None
        self.drag_mode      = None
        self.drag_corner    = None
        self.drag_start     = None
        self.drag_box_start = None

    def reset_to_original(self):
        """Restore all boxes to the model's original predictions."""
        self.boxes          = [dict(b) for b in self.orig_boxes]
        self.selected_idx   = None
        self.drag_mode      = None
        self.drag_start     = None

    def clear_all(self):
        """Remove every box — useful for clean / all-FP frames."""
        self.boxes          = []
        self.selected_idx   = None
        self.drag_mode      = None
        self.drag_start     = None

    def delete_selected(self):
        """Delete the currently selected box."""
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.boxes):
            del self.boxes[self.selected_idx]
            self.selected_idx = None
            self.drag_mode    = None
            self.drag_start   = None

    def cancel_drawing(self):
        """Cancel an in-progress new box drawing."""
        if self.drag_mode in ("new_drag", "new_click") and self.selected_idx is not None:
            if 0 <= self.selected_idx < len(self.boxes):
                del self.boxes[self.selected_idx]
        self.selected_idx = None
        self.drag_mode    = None
        self.drag_start   = None

    def undo_last(self):
        if self.drag_mode in ("new_drag", "new_click"):
            self.cancel_drawing()
            return
        if self.boxes:
            self.boxes.pop()
            self.selected_idx = None
            self.drag_mode    = None

    def _normalize_box(self, idx):
        if idx is not None and 0 <= idx < len(self.boxes):
            b = self.boxes[idx]
            x1, x2 = sorted((b["x1"], b["x2"]))
            y1, y2 = sorted((b["y1"], b["y2"]))
            b["x1"] = max(0, min(self.img_w - 1, x1))
            b["x2"] = max(0, min(self.img_w - 1, x2))
            b["y1"] = max(0, min(self.img_h - 1, y1))
            b["y2"] = max(0, min(self.img_h - 1, y2))
            # Discard degenerates (< 3px width or height)
            if b["x2"] - b["x1"] < 3 or b["y2"] - b["y1"] < 3:
                del self.boxes[idx]
                self.selected_idx = None

    def hit_box(self, x, y):
        """Return index of topmost box containing (x, y), or None."""
        for i in reversed(range(len(self.boxes))):
            b = self.boxes[i]
            x1, x2 = min(b["x1"], b["x2"]), max(b["x1"], b["x2"])
            y1, y2 = min(b["y1"], b["y2"]), max(b["y1"], b["y2"])
            if (x1 - BOX_HIT_MARGIN <= x <= x2 + BOX_HIT_MARGIN and
                    y1 - BOX_HIT_MARGIN <= y <= y2 + BOX_HIT_MARGIN):
                return i
        return None

    def hit_corner(self, box, x, y):
        x1, x2 = min(box["x1"], box["x2"]), max(box["x1"], box["x2"])
        y1, y2 = min(box["y1"], box["y2"]), max(box["y1"], box["y2"])
        corners = {
            0: (x1, y1), 1: (x2, y1),
            2: (x1, y2), 3: (x2, y2),
        }
        for idx, (cx, cy) in corners.items():
            if abs(cx - x) <= CORNER_GRAB_PX and abs(cy - y) <= CORNER_GRAB_PX:
                return idx
        return None

    def mouse_callback(self, event, x, y, flags, param):
        # Convert from display-pixel space back to original-image-pixel space
        x = int(round(x / max(self.scale_x, 1e-6)))
        y = int(round(y / max(self.scale_y, 1e-6)))
        x = max(0, min(self.img_w - 1, x))
        y = max(0, min(self.img_h - 1, y))

        # ── Right-click: delete clicked box OR cancel in-progress drawing ────
        if event == cv2.EVENT_RBUTTONDOWN:
            if self.drag_mode in ("new_click", "new_drag"):
                self.cancel_drawing()
                return
            idx = self.hit_box(x, y)
            if idx is not None:
                del self.boxes[idx]
                self.selected_idx = None
                self.drag_mode    = None
            return

        # ── Left-button down: start new box OR select/move/resize existing ──
        if event == cv2.EVENT_LBUTTONDOWN:
            # If in 2-click drawing mode, finalize the box!
            if self.drag_mode == "new_click" and self.selected_idx is not None:
                if 0 <= self.selected_idx < len(self.boxes):
                    b = self.boxes[self.selected_idx]
                    b["x2"], b["y2"] = x, y
                    self._normalize_box(self.selected_idx)
                self.drag_mode = None
                return

            # Shift key forces drawing a new box even if clicking over an existing box
            shift_pressed = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
            idx = None if shift_pressed else self.hit_box(x, y)

            if idx is not None:
                box    = self.boxes[idx]
                corner = self.hit_corner(box, x, y)
                self.selected_idx   = idx
                self.drag_start     = (x, y)
                self.drag_box_start = dict(box)
                self.drag_mode      = "resize" if corner is not None else "move"
                self.drag_corner    = corner
            else:
                # Start new box (class 0 — litter)
                self.selected_idx = None
                self.drag_mode    = "new_drag"
                self.drag_start   = (x, y)
                self.boxes.append({"cls": 0, "x1": x, "y1": y, "x2": x, "y2": y, "conf": None})
                self.selected_idx = len(self.boxes) - 1

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_mode in ("new_drag", "new_click") and self.selected_idx is not None:
                if 0 <= self.selected_idx < len(self.boxes):
                    b = self.boxes[self.selected_idx]
                    b["x1"], b["y1"] = self.drag_start
                    b["x2"], b["y2"] = x, y
            elif self.drag_mode == "move" and self.selected_idx is not None:
                if 0 <= self.selected_idx < len(self.boxes):
                    dx = x - self.drag_start[0]
                    dy = y - self.drag_start[1]
                    s  = self.drag_box_start
                    b  = self.boxes[self.selected_idx]
                    b["x1"] = max(0, min(self.img_w - 1, s["x1"] + dx))
                    b["y1"] = max(0, min(self.img_h - 1, s["y1"] + dy))
                    b["x2"] = max(0, min(self.img_w - 1, s["x2"] + dx))
                    b["y2"] = max(0, min(self.img_h - 1, s["y2"] + dy))
            elif self.drag_mode == "resize" and self.selected_idx is not None:
                if 0 <= self.selected_idx < len(self.boxes):
                    b = self.boxes[self.selected_idx]
                    if self.drag_corner in (0, 2):
                        b["x1"] = x
                    else:
                        b["x2"] = x
                    if self.drag_corner in (0, 1):
                        b["y1"] = y
                    else:
                        b["y2"] = y

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag_mode == "new_drag" and self.selected_idx is not None:
                dx = abs(x - self.drag_start[0])
                dy = abs(y - self.drag_start[1])
                if dx < 4 and dy < 4:
                    # User clicked without dragging -> switch to 2-click mode!
                    # The box follows the mouse until the second click.
                    self.drag_mode = "new_click"
                else:
                    self._normalize_box(self.selected_idx)
                    self.drag_mode = None
            elif self.drag_mode in ("move", "resize"):
                if self.selected_idx is not None:
                    self._normalize_box(self.selected_idx)
                self.drag_mode   = None
                self.drag_corner = None

    def render(self, frame, img_index, img_total, fname):
        """Draw boxes, selection handles, and the HUD bar onto a copy of frame."""
        vis = frame.copy()

        for i, b in enumerate(self.boxes):
            is_sel    = (i == self.selected_idx)
            color     = SEL_COLOR if is_sel else BOX_COLOR
            thickness = SEL_THICKNESS if is_sel else BOX_THICKNESS

            x1, y1 = min(b["x1"], b["x2"]), min(b["y1"], b["y2"])
            x2, y2 = max(b["x1"], b["x2"]), max(b["y1"], b["y2"])

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

            label = "litter" if b["conf"] is None else f"litter {b['conf']:.2f}"
            lx = max(x1, 2)
            ly = max(y1 - 6, 14)
            cv2.putText(vis, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # Resize corner handles when selected
            if is_sel:
                for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                    cv2.circle(vis, (cx, cy), 6, CORNER_COLOR, -1)
                    cv2.circle(vis, (cx, cy), 7, (0, 0, 0), 1)

        # ── HUD bar ──────────────────────────────────────────────────────────
        n_boxes = len(self.boxes)
        box_str = f"{n_boxes} box{'es' if n_boxes != 1 else ''}"

        sel_hint = ""
        if self.drag_mode == "new_click":
            sel_hint = "  |  [CLICK TO SET 2nd CORNER or R-click/Esc to cancel]"
        elif self.selected_idx is not None:
            sel_hint = "  |  [Del/d/Backspace] delete selected"

        hud_left  = f"[{img_index}/{img_total}] {fname}   {box_str}{sel_hint}"
        hud_right = ("s/Enter=save  n/Spc=skip  "
                     "Del/d/Rclick=delete  c=clear all  z=undo  u=reset  q=quit")

        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (vis.shape[1], HUD_HEIGHT), (15, 15, 15), -1)
        vis = cv2.addWeighted(overlay, 0.65, vis, 0.35, 0)

        cv2.putText(vis, hud_left,  (8, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 230, 200), 1, cv2.LINE_AA)
        cv2.putText(vis, hud_right, (8, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)
        return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args        = parse_args()
    class_names = [c.strip() for c in args.classes.split(",") if c.strip()] or ["litter"]

    if not os.path.isdir(args.images_dir):
        os.makedirs(args.images_dir, exist_ok=True)
        print(f"Created images directory: {args.images_dir}")

    out_images = os.path.join(args.output, "images")
    out_labels = os.path.join(args.output, "labels")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)
    state_path = os.path.join(args.output, "review_state.json")

    state = {"accepted": [], "rejected": []}
    if os.path.isfile(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            print(f"Warning: could not read existing state file ({e}), starting fresh.")
    done = set(state["accepted"]) | set(state["rejected"])

    # ── Load model ───────────────────────────────────────────────────────────
    model = None
    if args.weights and os.path.isfile(args.weights):
        try:
            from detector import YOLO
            print(f"Using lightweight detector.py backend: {args.weights}")
        except ImportError:
            from ultralytics import YOLO
            print(f"Using Ultralytics backend: {args.weights}")
        model = YOLO(args.weights)
    elif not args.preds_dir:
        fallback_onnx = os.path.join(_SWSTP_EDGE_DIR, "weights", "best.onnx")
        if os.path.isfile(fallback_onnx):
            try:
                from detector import YOLO
                print(f"Falling back to production ONNX: {fallback_onnx}")
                model = YOLO(fallback_onnx)
            except ImportError:
                sys.exit(f"Weights file not found: {args.weights}")
        else:
            sys.exit(f"Weights not found: {args.weights}. Provide --weights or --preds-dir")

    # ── Collect pending files ─────────────────────────────────────────────────
    files = sorted(f for f in os.listdir(args.images_dir)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS)
    if not files:
        print(f"\nNo images found in: {args.images_dir}")
        print("Place images (.jpg, .png, etc.) there to start reviewing.\n")
        return

    todo = [f for f in files if f not in done]
    print(f"\n[REVIEW] {len(files)} total  |  {len(todo)} to review  |  {len(done)} already done")
    print("Controls: s/Enter=save  n/Space=skip  Del/d/Rclick=delete  c=clear all  u=reset  q=quit\n")

    # ── Pre-run inference on all pending images ───────────────────────────────
    pred_cache = {}
    if model is not None and todo:
        print(f"Pre-running inference on {len(todo)} images (please wait)...")
        for i, fname in enumerate(todo, 1):
            img_path  = os.path.join(args.images_dir, fname)
            frame_tmp = imread_unicode(img_path)
            if frame_tmp is None:
                pred_cache[fname] = []
                print(f"  [{i}/{len(todo)}] {fname} — UNREADABLE, skipping")
                continue
            h_tmp, w_tmp = frame_tmp.shape[:2]
            results  = model.predict(frame_tmp, conf=args.conf, verbose=False)
            r        = results[0]
            boxes_tmp = []
            if r.boxes is not None:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    conf = float(b.conf.item())
                    boxes_tmp.append({"cls": 0, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf})
            pred_cache[fname] = boxes_tmp
            print(f"  [{i}/{len(todo)}] {fname} — {len(boxes_tmp)} detection(s)")
        print("Done. Opening review window...\n")

    # ── Interactive review loop ───────────────────────────────────────────────
    session = ReviewSession()
    window  = "Active Learning Review"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, session.mouse_callback)

    idx = 0
    while idx < len(todo):
        fname    = todo[idx]
        img_path = os.path.join(args.images_dir, fname)
        frame    = imread_unicode(img_path)
        if frame is None:
            print(f"[skip-unreadable] {fname}")
            idx += 1
            continue
        h, w = frame.shape[:2]

        # Load boxes for this frame
        if args.preds_dir:
            txt_path = os.path.join(args.preds_dir, os.path.splitext(fname)[0] + ".txt")
            boxes    = load_yolo_txt(txt_path, w, h)
        else:
            if fname in pred_cache:
                boxes = pred_cache[fname]
            else:
                results = model.predict(frame, conf=args.conf, verbose=False)
                r       = results[0]
                boxes   = []
                if r.boxes is not None:
                    for b in r.boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        conf = float(b.conf.item())
                        boxes.append({"cls": 0, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf})

        session.load(boxes, w, h)

        done_with_frame = False
        while not done_with_frame:
            vis = session.render(frame, idx + 1, len(todo), fname)
            display_vis, scale_x, scale_y = fit_frame(vis)
            session.scale_x = scale_x
            session.scale_y = scale_y
            cv2.imshow(window, display_vis)
            raw = cv2.waitKey(20)

            # CRITICAL: When no key is pressed, waitKey returns -1.
            # (-1 & 0xFF) evaluates to 255. Do not treat 255 as a key press!
            if raw == -1:
                continue

            key = raw & 0xFF

            # ── Quit ─────────────────────────────────────────────────────────
            if key in (ord("q"), 27):
                if session.drag_mode == "new_click":
                    session.cancel_drawing()
                    continue
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)
                cv2.destroyAllWindows()
                print(f"Quit. {len(state['accepted'])} accepted, {len(state['rejected'])} skipped.")
                return

            # ── Accept: save image + YOLO label ──────────────────────────────
            elif key in (ord("s"), 13):
                if session.drag_mode == "new_click":
                    session.cancel_drawing()
                save_yolo_txt(
                    os.path.join(out_labels, os.path.splitext(fname)[0] + ".txt"),
                    session.boxes, w, h,
                )
                imwrite_unicode(os.path.join(out_images, fname), frame)
                state["accepted"].append(fname)
                kept = len(session.boxes)
                orig = len(session.orig_boxes)
                removed = orig - kept
                msg = f"[save] {fname}  {kept} box(es)"
                if removed > 0:
                    msg += f"  ({removed} false positive(s) removed)"
                print(msg)
                done_with_frame = True

            # ── Skip (reject, not saved) ──────────────────────────────────────
            elif key in (ord("n"), ord(" ")):
                if session.drag_mode == "new_click":
                    session.cancel_drawing()
                state["rejected"].append(fname)
                print(f"[skip] {fname}")
                done_with_frame = True

            # ── Delete selected box (keyboard shortcut) ───────────────────────
            # Supports 'd', 'x', Backspace (8/127), and Windows Delete key
            elif key in (ord("d"), ord("x"), 8, 127) or raw in (0x2E0000, 3014656, 46) or key == 0:
                session.delete_selected()

            # ── Clear ALL boxes ───────────────────────────────────────────────
            elif key == ord("c"):
                session.clear_all()
                print(f"[clear-all] {fname} — all boxes removed")

            # ── Undo last added box ───────────────────────────────────────────
            elif key == ord("z"):
                session.undo_last()

            # ── Reset to original model predictions ───────────────────────────
            elif key == ord("u"):
                session.reset_to_original()

        idx += 1
        # Autosave progress every 10 frames
        if idx % 10 == 0:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    cv2.destroyAllWindows()
    print(f"\nReview complete. {len(state['accepted'])} saved, {len(state['rejected'])} skipped -> {args.output}")


if __name__ == "__main__":
    main()
