"""
litter_engine.py — Embedded Litter Detection Engine for SWSTP Unified.

Integrates YOLO inference into the motion detection loop WITHOUT opening
a second camera. Frames are passed in from the already-read main loop frame.

Trigger logic:
  - Every LITTER_DISTANCE_INTERVAL_M meters (default: 10 m) of GNSS travel,
    the current live frame is queued for YOLO inference.
  - If GNSS is lost for > LITTER_GNSS_LOST_TIMEOUT_SEC (default: 15 s),
    falls back to a clock-based trigger every LITTER_TIME_FALLBACK_SEC (default: 30 s).
  - On GNSS recovery, re-anchors to current position and resumes distance mode.

Inference:
  - Runs in a dedicated background daemon thread (never blocks the motion loop).
  - queue.Queue(maxsize=1): if inference is still running when the next trigger
    fires, the new frame is dropped (prevents queue buildup on slow hardware).
  - Saves annotated image + JSON sidecar to LITTER_CAPTURES_DIR only when
    litter is actually detected (not for clean frames).
  - Fires leds.notify_litter_snap() (Red-Yellow-Red-Yellow) on litter detection.

Area of Disinterest (AoD):
  - The ROI polygon used by motion detection (roi_polygon.json) is inverted here:
    the same polygon area is treated as an Area of Disinterest.
  - Detections whose bounding box overlaps the AoD polygon by >= LITTER_OVERLAP_THRESHOLD
    are silently ignored (e.g. the vehicle's own hood / bonnet).
"""

import datetime
import json
import os
import queue
import threading
import time

import cv2
import numpy as np

# Try lightweight ONNX detector first, fall back to ultralytics
try:
    from detector import YOLO
except ImportError:
    from ultralytics import YOLO  # type: ignore

import config as _cfg
import sensors.gnss as gnss_sensor
import sensors.leds as leds


def _haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two coordinates in meters."""
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return 0.0
    try:
        import math
        R = 6371000.0
        p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
        dp = math.radians(float(lat2) - float(lat1))
        dl = math.radians(float(lon2) - float(lon1))
        a = math.sin(dp / 2.0)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0)**2
        return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _polygon_overlap(box, poly_pts, frame_shape) -> float:
    """Fraction of the bounding-box that lies inside the polygon AoD."""
    if poly_pts is None or len(poly_pts) < 3:
        return 0.0
    x1, y1, x2, y2 = box
    h, w = frame_shape[:2]
    bx1, by1 = max(0, int(x1)), max(0, int(y1))
    bx2, by2 = min(w, int(x2)), min(h, int(y2))
    if bx2 <= bx1 or by2 <= by1:
        return 0.0
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(poly_pts, dtype=np.int32)], 255)
    roi = mask[by1:by2, bx1:bx2]
    box_area = (bx2 - bx1) * (by2 - by1)
    return float(np.count_nonzero(roi)) / box_area if box_area > 0 else 0.0


def _build_pixel_aod(poly_pts_normalized, frame_w: int, frame_h: int):
    """
    Convert normalised ROI polygon (0.0–1.0) to pixel coordinates for AoD use.
    Returns a list of (x, y) integer tuples, or [] if invalid.
    """
    if not poly_pts_normalized or len(poly_pts_normalized) < 3:
        return []
    result = []
    for p in poly_pts_normalized:
        px = int(p[0] * frame_w) if p[0] <= 1.0 else int(p[0])
        py = int(p[1] * frame_h) if p[1] <= 1.0 else int(p[1])
        result.append((px, py))
    return result


# ---------------------------------------------------------------------------
# Async disk writer (zero save latency impact on inference thread)
# ---------------------------------------------------------------------------
_save_queue: queue.Queue = queue.Queue()


def _save_worker() -> None:
    """Background daemon: writes JPEG + JSON sidecar to disk."""
    while True:
        item = _save_queue.get()
        if item is None:
            _save_queue.task_done()
            break
        filename, final_frame, sidecar_filename, sidecar_data = item
        try:
            cv2.imwrite(filename, final_frame)
            with open(sidecar_filename, "w", encoding="utf-8") as sf:
                json.dump(sidecar_data, sf, indent=2)
            print(f"[LITTER] Saved {filename} ({final_frame.shape[1]}x{final_frame.shape[0]})")

            # Real-time upload: enqueue into upload_queue immediately after disk write
            try:
                from telemetry import upload_queue
                litter_lat = sidecar_data.get("gnss", {}).get("latitude")
                litter_lon = sidecar_data.get("gnss", {}).get("longitude")
                upload_item = {
                    "local_path":           filename,
                    "meta_path":            sidecar_filename,
                    "captured_at":          sidecar_data.get("captured_at"),
                    "latitude":             litter_lat,
                    "longitude":            litter_lon,
                    "detection_type":       "LITTER",
                    "detection_confidence": sidecar_data.get("detection_confidence", 0.0),
                    "stop_duration_sec":    None,
                }
                upload_queue.put_nowait(upload_item)
            except Exception as q_err:
                print(f"[LITTER] Note: queued via local disk backup ({q_err})")

        except Exception as e:
            print(f"[LITTER] Error writing {filename}: {e}")
        finally:
            _save_queue.task_done()


_save_thread: threading.Thread | None = None


def _ensure_save_thread() -> None:
    global _save_thread
    if _save_thread is None or not _save_thread.is_alive():
        _save_thread = threading.Thread(target=_save_worker, daemon=True, name="litter-save-worker")
        _save_thread.start()


# ---------------------------------------------------------------------------
# LitterEngine
# ---------------------------------------------------------------------------

class LitterEngine:
    """
    Encapsulates YOLO litter detection triggered by GNSS distance travel.

    Usage:
        engine = LitterEngine(...)          # Load model once at startup
        # In main loop, every frame:
        if engine.update_gnss(lat, lon):   # Returns True every 10 m
            engine.try_trigger(frame, lat, lon, gnss_data, rtc_ts)
        # On shutdown:
        engine.shutdown()
    """

    def __init__(
        self,
        weights_path: str,
        output_dir: str,
        conf: float = _cfg.LITTER_CONF_THRESHOLD,
        overlap_threshold: float = _cfg.LITTER_OVERLAP_THRESHOLD,
        aod_polygon_normalized: list | None = None,
        distance_interval_m: float = _cfg.LITTER_DISTANCE_INTERVAL_M,
        gnss_lost_timeout_sec: float = _cfg.LITTER_GNSS_LOST_TIMEOUT_SEC,
        time_fallback_sec: float = _cfg.LITTER_TIME_FALLBACK_SEC,
        device_id: str = "UNPROVISIONED",
    ):
        self.output_dir           = output_dir
        self.conf                 = conf
        self.overlap_threshold    = overlap_threshold
        self.aod_polygon_norm     = aod_polygon_normalized  # normalized [0-1] coords (from roi_polygon.json)
        self.distance_interval_m  = distance_interval_m
        self.gnss_lost_timeout_sec = gnss_lost_timeout_sec
        self.time_fallback_sec    = time_fallback_sec
        self.device_id            = device_id

        os.makedirs(output_dir, exist_ok=True)
        _ensure_save_thread()

        # GNSS distance tracking state
        self._op_mode              = "GNSS_DISTANCE"  # or "TIME_FALLBACK"
        self._last_trigger_lat     = None
        self._last_trigger_lon     = None
        self._gnss_loss_start      = None             # monotonic time when fix was lost
        self._last_fallback_time   = time.monotonic()
        self._total_distance_m     = 0.0
        self._distance_since_trig  = 0.0
        self._trigger_count        = 0
        self._event_count          = 0                # number of frames where litter was saved
        self._gnss_initialized     = False            # True once we have first valid fix

        # Per-frame inference queue (maxsize=1 = drop new if busy)
        self._infer_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event              = threading.Event()

        # Load model in a background thread so the main loop isn't blocked
        self._model     = None
        self._model_ready = threading.Event()
        self._model_load_thread = threading.Thread(
            target=self._load_model,
            args=(weights_path,),
            name="litter-model-load",
            daemon=True,
        )
        self._model_load_thread.start()

        # Start inference worker thread
        self._infer_thread = threading.Thread(
            target=self._inference_worker,
            name="litter-infer",
            daemon=True,
        )
        self._infer_thread.start()

        print(f"[LITTER] Engine initializing — weights: {weights_path}")
        print(f"[LITTER] Distance trigger: {distance_interval_m:.1f}m | "
              f"GNSS timeout: {gnss_lost_timeout_sec:.0f}s | "
              f"Clock fallback: {time_fallback_sec:.0f}s | "
              f"Output: {output_dir}")

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------
    def _load_model(self, weights_path: str) -> None:
        try:
            self._model = YOLO(weights_path)
            self._model_ready.set()
            print(f"[LITTER] YOLO model loaded and ready.")
        except Exception as exc:
            print(f"[LITTER] ERROR loading YOLO model: {exc}")
            # model stays None; inference worker will skip gracefully

    # -------------------------------------------------------------------------
    # GNSS distance tracking
    # -------------------------------------------------------------------------
    def update_gnss(self, lat, lon) -> bool:
        """
        Call once per main loop iteration with current GNSS coordinates.

        Returns True when the 10 m distance threshold is crossed (trigger condition).
        Handles GNSS loss / recovery and clock fallback automatically.
        """
        now = time.monotonic()
        has_fix = (lat is not None and lon is not None
                   and not (abs(lat) < 0.001 and abs(lon) < 0.001))

        # ── GNSS mode transitions ─────────────────────────────────────────
        if has_fix:
            if not self._gnss_initialized:
                # First fix — anchor starting position
                self._last_trigger_lat = lat
                self._last_trigger_lon = lon
                self._op_mode = "GNSS_DISTANCE"
                self._gnss_initialized = True
                self._gnss_loss_start  = None
                print(f"[LITTER TRIGGER] GNSS distance mode anchored at "
                      f"({lat:.8f}, {lon:.8f}). Trigger every {self.distance_interval_m:.1f}m.")
            elif self._op_mode == "TIME_FALLBACK":
                # Regained fix — re-anchor
                self._last_trigger_lat = lat
                self._last_trigger_lon = lon
                self._op_mode          = "GNSS_DISTANCE"
                self._gnss_loss_start  = None
                print(f"[LITTER TRIGGER] GNSS fix regained @ ({lat:.8f}, {lon:.8f}). "
                      f"Re-anchoring to {self.distance_interval_m:.1f}m distance mode.")
            elif self._gnss_loss_start is not None:
                # Recovered within grace period
                self._gnss_loss_start = None
        else:
            # No fix
            if self._op_mode == "GNSS_DISTANCE":
                if self._gnss_loss_start is None:
                    self._gnss_loss_start = now
                    print(f"[LITTER TRIGGER] GNSS lost. Grace period: {self.gnss_lost_timeout_sec:.0f}s ...")
                elif (now - self._gnss_loss_start) >= self.gnss_lost_timeout_sec:
                    self._op_mode          = "TIME_FALLBACK"
                    self._last_fallback_time = now
                    self._gnss_loss_start  = None
                    print(f"[LITTER TRIGGER] GNSS lost >{self.gnss_lost_timeout_sec:.0f}s. "
                          f"Switching to clock fallback (every {self.time_fallback_sec:.0f}s).")

        # ── Evaluate trigger condition ─────────────────────────────────────
        trigger = False

        if self._op_mode == "GNSS_DISTANCE" and has_fix and self._last_trigger_lat is not None:
            dist = _haversine_distance_m(
                self._last_trigger_lat, self._last_trigger_lon, lat, lon
            )
            self._distance_since_trig = dist
            if dist >= self.distance_interval_m:
                self._total_distance_m  += dist
                self._last_trigger_lat   = lat
                self._last_trigger_lon   = lon
                self._distance_since_trig = 0.0
                trigger = True

        elif self._op_mode == "TIME_FALLBACK":
            elapsed = now - self._last_fallback_time
            if elapsed >= self.time_fallback_sec:
                self._last_fallback_time = now
                trigger = True

        return trigger

    # -------------------------------------------------------------------------
    # Non-blocking trigger
    # -------------------------------------------------------------------------
    def try_trigger(self, frame, lat, lon, gnss_data: dict, rtc_ts: str | None) -> None:
        """
        Post a frame for litter inference.  Non-blocking — if a previous inference
        is still running the frame is dropped (queue maxsize=1 prevents buildup).

        Args:
            frame:     BGR ndarray copied from main loop (not the live cap buffer).
            lat, lon:  Current GNSS coordinates at time of trigger.
            gnss_data: Dict with live GNSS fields (speed, satellites, etc.).
            rtc_ts:    RTC ISO timestamp string or None.
        """
        if not self._model_ready.is_set():
            # Model still loading — silently skip this trigger
            return

        job = {
            "frame":      frame,
            "lat":        lat,
            "lon":        lon,
            "gnss_data":  gnss_data,
            "rtc_ts":     rtc_ts,
            "trigger_n":  self._trigger_count + 1,
            "op_mode":    self._op_mode,
            "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        try:
            self._infer_queue.put_nowait(job)
            self._trigger_count += 1
            print(f"[LITTER] Trigger #{self._trigger_count} queued "
                  f"({self._op_mode}, dist: {self._distance_since_trig:.1f}m)")
            try:
                leds.notify_litter_capture()
            except Exception:
                pass
        except queue.Full:
            # Previous inference still running — drop this frame
            print(f"[LITTER] Trigger skipped — inference busy (previous frame still processing).")

    # -------------------------------------------------------------------------
    # Background inference worker
    # -------------------------------------------------------------------------
    def _inference_worker(self) -> None:
        """Daemon thread: dequeues frames, runs YOLO, saves results."""
        while not self._stop_event.is_set():
            try:
                job = self._infer_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._run_inference(job)
            except Exception as exc:
                print(f"[LITTER] Inference error: {exc}")
            finally:
                self._infer_queue.task_done()

    def _run_inference(self, job: dict) -> None:
        frame      = job["frame"]
        lat        = job["lat"]
        lon        = job["lon"]
        gnss_data  = job["gnss_data"]
        rtc_ts     = job["rtc_ts"]
        trigger_n  = job["trigger_n"]
        op_mode    = job["op_mode"]
        captured_at = job["captured_at"]

        if self._model is None:
            print(f"[LITTER] Trigger #{trigger_n}: model not loaded, skipping.")
            return

        h, w = frame.shape[:2]

        # Build pixel-coordinate AoD polygon from normalised ROI
        aod_pts_px = _build_pixel_aod(self.aod_polygon_norm, w, h) if self.aod_polygon_norm else []

        # ── Run YOLO inference ─────────────────────────────────────────────
        t_start = time.monotonic()
        try:
            results = self._model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.conf,
                verbose=False,
            )
        except Exception as exc:
            print(f"[LITTER] Trigger #{trigger_n}: YOLO error: {exc}")
            return
        infer_ms = int((time.monotonic() - t_start) * 1000)

        result = results[0]
        save_canvas = frame.copy()
        detected_items = []
        has_litter = False

        if result.boxes is not None:
            for box in result.boxes:
                if box.id is None:
                    continue
                track_id       = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_score     = float(box.conf[0])
                cls_id         = int(box.cls[0])
                cls_name       = self._model.names.get(cls_id, str(cls_id))

                # Check AoD overlap (same polygon as motion ROI, inverted use)
                in_aod = False
                if aod_pts_px:
                    in_aod = (_polygon_overlap((x1, y1, x2, y2), aod_pts_px, frame.shape)
                              >= self.overlap_threshold)

                detected_items.append({
                    "track_id":   track_id,
                    "class_id":   cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf_score, 4),
                    "bbox":       [x1, y1, x2, y2],
                    "in_aod":     in_aod,
                })

                if in_aod:
                    continue  # Ignore detections inside AoD

                has_litter = True
                label = f"{cls_name} #{track_id}  {conf_score:.2f}"
                cv2.rectangle(save_canvas, (x1, y1), (x2, y2), (0, 230, 0), 3)
                cv2.putText(save_canvas, label, (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 0), 2, cv2.LINE_AA)

        total_det = len([d for d in detected_items if not d["in_aod"]])
        print(f"[LITTER] Trigger #{trigger_n}: {total_det} litter detected "
              f"({len(detected_items)} total, {infer_ms}ms inference)")

        # ── Save only if litter was detected ──────────────────────────────
        if has_litter:
            self._event_count += 1

            # Bottom banner (non-destructive, appended below image)
            has_fix = (lat is not None and lon is not None
                       and not (abs(float(lat or 0)) < 0.001 and abs(float(lon or 0)) < 0.001))
            strip_h = 70
            banner  = np.zeros((strip_h, w, 3), dtype=np.uint8)
            cv2.line(banner, (0, 0), (w, 0), (50, 50, 50), 1)

            # RTC timestamp
            rtc_display = rtc_ts if rtc_ts else captured_at[:19].replace("T", " ")
            cv2.putText(banner, f"RTC: {rtc_display}",
                        (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            # GNSS
            if has_fix:
                gps_text = f"GNSS: {float(lat):.8f}, {float(lon):.8f}"
                gps_col  = (0, 255, 0)
            elif op_mode == "TIME_FALLBACK":
                gps_text = "GNSS: NO FIX (CLOCK TRIGGER)"
                gps_col  = (0, 165, 255)
            else:
                gps_text = "GNSS: NO FIX"
                gps_col  = (0, 0, 255)
            cv2.putText(banner, gps_text,
                        (14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, gps_col, 2, cv2.LINE_AA)

            final_frame = np.vstack([save_canvas, banner])

            # Safe filename: litter_<trigger>_<event>_<timestamp>.jpg
            clean_ts    = captured_at[:19].replace(":", "-").replace("T", "_")
            filename    = os.path.join(self.output_dir,
                                       f"litter_T{trigger_n:04d}_E{self._event_count:04d}_{clean_ts}.jpg")
            sidecar_fn  = os.path.splitext(filename)[0] + ".json"

            sidecar_data = {
                "event_number":   self._event_count,
                "trigger_number": trigger_n,
                "image_file":     os.path.basename(filename),
                "device_id":      self.device_id,
                "captured_at":    captured_at,
                "trigger_mode":   op_mode,
                "rtc_timestamp":  rtc_ts,
                "inference_ms":   infer_ms,
                "gnss": {
                    "fix":        has_fix,
                    "latitude":   float(lat) if lat is not None else None,
                    "longitude":  float(lon) if lon is not None else None,
                    "speed_kmh":  gnss_data.get("speed") if isinstance(gnss_data, dict) else None,
                    "satellites": gnss_data.get("satellites") if isinstance(gnss_data, dict) else None,
                },
                "detections":          detected_items,
                "detection_type":      "LITTER",
                "detection_confidence": round(
                    max((d["confidence"] for d in detected_items if not d.get("in_aod")), default=0.0), 4
                ),
            }

            # Async write (does not block inference thread beyond queue put)
            _save_queue.put((filename, final_frame, sidecar_fn, sidecar_data))

            # LED: Red-Yellow-Red-Yellow blink — litter detected!
            try:
                leds.notify_litter_snap()
            except Exception:
                pass

            print(f"[LITTER] *** LITTER DETECTED *** Event #{self._event_count} → {filename}")
        else:
            # Clean frame — no save, no LED blink
            print(f"[LITTER] Trigger #{trigger_n}: clean frame (0 litter). No save.")

    # -------------------------------------------------------------------------
    # Status / HUD info
    # -------------------------------------------------------------------------
    @property
    def op_mode(self) -> str:
        return self._op_mode

    @property
    def distance_since_trigger(self) -> float:
        return self._distance_since_trig

    @property
    def total_distance_m(self) -> float:
        return self._total_distance_m

    @property
    def trigger_count(self) -> int:
        return self._trigger_count

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def model_ready(self) -> bool:
        return self._model_ready.is_set()

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------
    def shutdown(self) -> None:
        """Gracefully stop inference worker and flush pending disk writes."""
        print("[LITTER] Shutting down — waiting for pending inference & saves...")
        self._stop_event.set()
        try:
            self._infer_queue.join()
        except Exception:
            pass
        # Drain the save queue
        try:
            _save_queue.join()
        except Exception:
            pass
        print("[LITTER] Engine shutdown complete.")


# ---------------------------------------------------------------------------
# Cleanup helper (mirrors motion.py cleanup_old_local_captures)
# ---------------------------------------------------------------------------
_last_litter_cleanup = 0.0


def cleanup_old_litter_captures(save_dir: str, retention_days: int = 3) -> None:
    """Safety guard: delete litter captures older than retention_days (rate-limited to 1/min)."""
    global _last_litter_cleanup
    now = time.time()
    if (now - _last_litter_cleanup) < 60.0:
        return
    _last_litter_cleanup = now
    if not os.path.exists(save_dir):
        return
    cutoff = now - (retention_days * 86400)
    try:
        with os.scandir(save_dir) as entries:
            for entry in entries:
                if entry.is_file():
                    try:
                        if entry.stat().st_mtime < cutoff:
                            os.remove(entry.path)
                            if entry.name.lower().endswith(".jpg"):
                                json_path = os.path.splitext(entry.path)[0] + ".json"
                                if os.path.exists(json_path):
                                    os.remove(json_path)
                    except Exception:
                        pass
    except Exception:
        pass
