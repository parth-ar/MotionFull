"""
detector.py — Lightweight, Zero-PyTorch YOLOv8 Inference & ByteTrack Tracking.
------------------------------------------------------------------------------
Designed for Raspberry Pi (Python 3.14 / Linux / Windows) to replace bulky
PyTorch/Ultralytics dependencies with high-performance OpenCV DNN / ONNX Runtime.

Features:
- Pure NumPy + OpenCV runtime (zero PyTorch, zero TorchVision, zero Ultralytics).
- Transparent drop-in YOLO(...) API compatible with litter_event_logger and review_app.
- Automatic backend selection: uses onnxruntime if installed; falls back to cv2.dnn.
- Automatic weight resolution: if a .pt path is given without PyTorch installed,
  automatically uses the sibling .onnx or _int8.onnx file.
- Built-in pure-NumPy ByteTrack tracker (Kalman filter + 2-stage IoU association).
"""

import os
import sys
import numpy as np
import cv2

# Check for ONNX Runtime
try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Letterbox image preprocessing (standard YOLOv8 letterboxing)
# ---------------------------------------------------------------------------

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize image and pad to new_shape maintaining aspect ratio."""
    shape = im.shape[:2]  # [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2.0
    dh /= 2.0

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im_padded = cv2.copyMakeBorder(
        im_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    return im_padded, r, (dw, dh)


# ---------------------------------------------------------------------------
# Pure NumPy Kalman Filter for Bounding Box Tracking [x, y, a, h]
# ---------------------------------------------------------------------------

class KalmanFilterBox:
    """Kalman filter for tracking bounding boxes in image space."""

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, i + ndim] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T
        )) + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean = np.dot(self._update_mat, mean)
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        projected_cov = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T
        )) + np.diag(np.square(std))

        chol_factor, lower = None, True
        try:
            import scipy.linalg
            chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
            kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        except Exception:
            kalman_gain = np.dot(
                np.dot(covariance, self._update_mat.T),
                np.linalg.pinv(projected_cov)
            )

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T
        ))
        return new_mean, new_covariance


# ---------------------------------------------------------------------------
# Track State & Management
# ---------------------------------------------------------------------------

class STrack:
    """Individual object track state."""
    _count = 0

    def __init__(self, tlwh, score, cls_id):
        STrack._count += 1
        self.track_id = STrack._count
        self.is_activated = False
        self.state = 1  # 1: Tracked, 2: Lost, 3: Removed
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = float(score)
        self.cls_id = int(cls_id)
        self.kalman_filter = KalmanFilterBox()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self.tlwh))
        self.tracklet_len = 0
        self.time_since_update = 0

    @staticmethod
    def reset_counter():
        STrack._count = 0

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh, dtype=float).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= max(ret[3], 1e-6)
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh, dtype=float).copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def tlbr(self):
        if self.mean is None:
            return self.tlwh_to_tlbr(self.tlwh)
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return self.tlwh_to_tlbr(ret)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != 1:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def update(self, new_track, frame_id):
        self.tracklet_len += 1
        self.tlwh = new_track.tlwh
        self.score = new_track.score
        self.cls_id = new_track.cls_id
        self.state = 1
        self.is_activated = True
        self.time_since_update = 0
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(self.tlwh)
        )

    def mark_lost(self):
        self.state = 2

    def mark_removed(self):
        self.state = 3


# ---------------------------------------------------------------------------
# Pure NumPy IoU Association & ByteTracker
# ---------------------------------------------------------------------------

def bbox_ious(atlbrs, btlbrs):
    """Compute IoU matrix between two sets of bounding boxes in tlbr format."""
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    for i, a in enumerate(atlbrs):
        a_area = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
        for j, b in enumerate(btlbrs):
            b_area = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
            xx1 = max(a[0], b[0])
            yy1 = max(a[1], b[1])
            xx2 = min(a[2], b[2])
            yy2 = min(a[3], b[3])
            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            inter = w * h
            union = a_area + b_area - inter
            ious[i, j] = inter / union if union > 0 else 0.0
    return ious


def linear_assignment(cost_matrix, thresh):
    """Simple, greedy/Hungarian assignment for bipartite matching."""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    try:
        import scipy.optimize
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    except Exception:
        # High-performance greedy fallback if scipy is not installed
        matched = []
        rows_done = set()
        cols_done = set()
        flat_indices = np.argsort(cost_matrix, axis=None)
        for idx in flat_indices:
            r = idx // cost_matrix.shape[1]
            c = idx % cost_matrix.shape[1]
            if r not in rows_done and c not in cols_done:
                if cost_matrix[r, c] <= thresh:
                    matched.append((r, c))
                    rows_done.add(r)
                    cols_done.add(c)
        matches = np.array(matched, dtype=int) if matched else np.empty((0, 2), dtype=int)
        unmatched_a = tuple(i for i in range(cost_matrix.shape[0]) if i not in rows_done)
        unmatched_b = tuple(j for j in range(cost_matrix.shape[1]) if j not in cols_done)
        return matches, unmatched_a, unmatched_b

    matches = []
    unmatched_a = []
    unmatched_b = list(range(cost_matrix.shape[1]))
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            if c in unmatched_b:
                unmatched_b.remove(c)
        else:
            unmatched_a.append(r)
    for r in range(cost_matrix.shape[0]):
        if r not in row_ind:
            unmatched_a.append(r)
    return np.array(matches, dtype=int) if matches else np.empty((0, 2), dtype=int), tuple(unmatched_a), tuple(unmatched_b)


class BYTETracker:
    """ByteTrack multi-object tracker."""

    def __init__(self, track_thresh=0.25, match_thresh=0.8, max_time_lost=30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        self.tracked_stracks = []
        self.lost_stracks = []
        self.frame_id = 0

    def update(self, output_results):
        """
        output_results: array of [x1, y1, x2, y2, score, cls_id]
        returns list of active STrack objects
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []

        if len(output_results) > 0:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            classes = output_results[:, 5]

            remain_inds = scores >= self.track_thresh
            inds_low = scores < self.track_thresh
            inds_high = remain_inds

            dets = [STrack([b[0], b[1], b[2] - b[0], b[3] - b[1]], s, c)
                    for b, s, c in zip(bboxes[inds_high], scores[inds_high], classes[inds_high])]
            dets_second = [STrack([b[0], b[1], b[2] - b[0], b[3] - b[1]], s, c)
                           for b, s, c in zip(bboxes[inds_low], scores[inds_low], classes[inds_low])]
        else:
            dets = []
            dets_second = []

        # Predict current locations of existing tracks
        strack_pool = [t for t in self.tracked_stracks if t.state == 1]
        strack_pool += [t for t in self.lost_stracks if t.state == 2]
        for s in strack_pool:
            s.predict()

        # 1st association with high score detections
        dists = 1.0 - bbox_ious([t.tlbr for t in strack_pool], [d.tlbr for d in dets])
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = dets[idet]
            if track.state == 1:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.update(det, self.frame_id)
                refind_stracks.append(track)

        # 2nd association with low score detections
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == 1]
        dists2 = 1.0 - bbox_ious([t.tlbr for t in r_tracked_stracks], [d.tlbr for d in dets_second])
        matches2, u_track2, _ = linear_assignment(dists2, thresh=0.5)

        for itracked, idet in matches2:
            track = r_tracked_stracks[itracked]
            det = dets_second[idet]
            if track.state == 1:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.update(det, self.frame_id)
                refind_stracks.append(track)

        for it in u_track2:
            track = r_tracked_stracks[it]
            if track.state != 2:
                track.mark_lost()

        # Initialize new tracks for unmatched high score detections
        for inew in u_detection:
            track = dets[inew]
            if track.score >= self.track_thresh:
                track.is_activated = True
                activated_stracks.append(track)

        # Update track states
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 1]
        for t in activated_stracks:
            if t not in self.tracked_stracks:
                self.tracked_stracks.append(t)
        for t in refind_stracks:
            if t not in self.tracked_stracks:
                self.tracked_stracks.append(t)

        for t in self.lost_stracks:
            if self.frame_id - t.frame_id > self.max_time_lost:
                t.mark_removed()
        self.lost_stracks = [t for t in self.lost_stracks if t.state == 2]

        # Output active tracks
        output_stracks = [t for t in self.tracked_stracks if t.is_activated]
        return output_stracks


# ---------------------------------------------------------------------------
# Drop-In Result & Box Classes (Mimics Ultralytics Results API 1:1)
# ---------------------------------------------------------------------------

class TrackIdWrapper(int):
    """Integer wrapper with .item() method for Ultralytics API compatibility."""
    def item(self):
        return int(self)


class DetectionBox:
    """Individual bounding box mimicking ultralytics.engine.results.Boxes."""

    def __init__(self, xyxy, conf, cls_id, track_id=None):
        self.xyxy = np.array([xyxy], dtype=np.float32)
        self.conf = np.array([float(conf)], dtype=np.float32)
        self.cls = np.array([int(cls_id)], dtype=np.float32)
        self.id = TrackIdWrapper(int(track_id)) if track_id is not None else None


class DetectionResult:
    """Container mimicking ultralytics Results object."""

    def __init__(self, boxes=None, orig_shape=None):
        self.boxes = boxes if (boxes is not None and len(boxes) > 0) else None
        self.orig_shape = orig_shape


# ---------------------------------------------------------------------------
# Drop-In Lightweight YOLO Class
# ---------------------------------------------------------------------------

class YOLO:
    """
    Drop-in replacement for ultralytics.YOLO.
    Loads ONNX weights and runs on OpenCV DNN or ONNX Runtime.
    """

    def __init__(self, weights_path):
        resolved_path = self._resolve_weights_path(weights_path)
        self.weights_path = resolved_path
        self.names = {0: "litter"}
        self.tracker = BYTETracker()
        self.input_shape = (640, 640)
        self.backend = None

        # Try ONNX Runtime first, then fall back to OpenCV DNN
        if _ORT_AVAILABLE:
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(resolved_path, opts, providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
                self.backend = "onnxruntime"
                print(f"[DETECTOR] Loaded {resolved_path} via ONNX Runtime (CPU, 4 threads)")
            except Exception as e:
                print(f"[DETECTOR] Note: ONNX Runtime load failed ({e}), falling back to cv2.dnn")
                self.session = None

        if self.backend is None:
            try:
                self.net = cv2.dnn.readNetFromONNX(resolved_path)
                try:
                    cv2.setNumThreads(4)
                except Exception:
                    pass
                self.backend = "cv2.dnn"
                print(f"[DETECTOR] Loaded {resolved_path} via OpenCV DNN")
            except Exception as e:
                # If int8 ONNX has unsupported quantize ops in cv2.dnn, fall back to standard float ONNX
                fallback_onnx = os.path.join(os.path.dirname(resolved_path) or "weights", "best.onnx")
                if os.path.normpath(resolved_path) != os.path.normpath(fallback_onnx) and os.path.isfile(fallback_onnx):
                    print(f"[DETECTOR] Note: {resolved_path} failed in OpenCV DNN ({e}). Falling back to {fallback_onnx}...")
                    self.net = cv2.dnn.readNetFromONNX(fallback_onnx)
                    try:
                        cv2.setNumThreads(4)
                    except Exception:
                        pass
                    self.backend = "cv2.dnn"
                    print(f"[DETECTOR] Loaded {fallback_onnx} via OpenCV DNN")
                else:
                    raise

    def _resolve_weights_path(self, path):
        """Resolve .pt paths to .onnx or _int8.onnx if .pt is requested without PyTorch."""
        if os.path.isfile(path) and path.lower().endswith(".onnx"):
            return path

        base, ext = os.path.splitext(path)
        candidates = [
            f"{base}_int8.onnx",
            f"{base}.onnx",
            path,
            os.path.join("weights", "best_int8.onnx"),
            os.path.join("weights", "best.onnx"),
        ]

        for cand in candidates:
            if os.path.isfile(cand) and cand.lower().endswith(".onnx"):
                if ext.lower() == ".pt":
                    print(f"[DETECTOR] Redirecting PyTorch weights '{path}' -> Lightweight ONNX '{cand}'")
                return cand

        if os.path.isfile(path):
            return path

        raise FileNotFoundError(f"Weights file not found: {path}")

    def _forward(self, frame, conf_thresh=0.25, iou_thresh=0.7):
        """Run preprocessing, model forward pass, and NMS."""
        h_orig, w_orig = frame.shape[:2]
        img_padded, ratio, (pad_w, pad_h) = letterbox(frame, new_shape=self.input_shape)

        # Normalize 0..1 RGB float32
        if self.backend == "onnxruntime":
            img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
            img_tensor = img_rgb.astype(np.float32) / 255.0
            img_tensor = np.transpose(img_tensor, (2, 0, 1))  # HWC -> CHW
            img_tensor = np.expand_dims(img_tensor, axis=0)    # CHW -> BCHW
            raw_out = self.session.run(None, {self.input_name: img_tensor})[0]
        else:
            blob = cv2.dnn.blobFromImage(
                img_padded, 1.0 / 255.0, self.input_shape,
                swapRB=True, crop=False
            )
            self.net.setInput(blob)
            raw_out = self.net.forward()

        # raw_out shape: (1, 4 + num_classes, 8400)
        preds = raw_out[0]
        boxes_cxcywh = preds[:4, :].T  # (8400, 4)
        class_scores = preds[4:, :].T  # (8400, num_classes)

        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        mask = max_scores >= conf_thresh
        if not np.any(mask):
            return []

        filt_boxes = boxes_cxcywh[mask]
        filt_scores = max_scores[mask]
        filt_classes = class_ids[mask]

        # Convert cx, cy, w, h to top-left x, y, w, h for OpenCV NMS
        nms_boxes = []
        for b in filt_boxes:
            cx, cy, w, h = b
            nms_boxes.append([float(cx - w / 2.0), float(cy - h / 2.0), float(w), float(h)])

        indices = cv2.dnn.NMSBoxes(nms_boxes, filt_scores.tolist(), conf_thresh, iou_thresh)
        if len(indices) == 0:
            return []

        detections = []
        for idx in indices:
            i = idx if isinstance(idx, (int, np.integer)) else idx[0]
            bx, by, bw, bh = nms_boxes[i]
            # Rescale back to original image coordinates
            x1 = max(0.0, min(float((bx - pad_w) / ratio), float(w_orig)))
            y1 = max(0.0, min(float((by - pad_h) / ratio), float(h_orig)))
            x2 = max(0.0, min(float((bx + bw - pad_w) / ratio), float(w_orig)))
            y2 = max(0.0, min(float((by + bh - pad_h) / ratio), float(h_orig)))
            score = float(filt_scores[i])
            cls_id = int(filt_classes[i])
            detections.append([x1, y1, x2, y2, score, cls_id])

        return detections

    def track(self, frame, persist=True, tracker="bytetrack.yaml", conf=0.25, verbose=False):
        """Run YOLO inference and ByteTrack tracking on a frame."""
        raw_dets = self._forward(frame, conf_thresh=conf)
        if not persist:
            self.tracker = BYTETracker()

        det_arr = np.array(raw_dets, dtype=np.float32) if len(raw_dets) > 0 else np.empty((0, 6), dtype=np.float32)
        online_targets = self.tracker.update(det_arr)

        boxes = []
        for t in online_targets:
            tlbr = t.tlbr
            boxes.append(DetectionBox(
                xyxy=[tlbr[0], tlbr[1], tlbr[2], tlbr[3]],
                conf=t.score,
                cls_id=t.cls_id,
                track_id=t.track_id
            ))

        return [DetectionResult(boxes=boxes, orig_shape=frame.shape[:2])]

    def __call__(self, frame, conf=0.25, verbose=False):
        """Standard forward pass without tracking (for review_app.py)."""
        raw_dets = self._forward(frame, conf_thresh=conf)
        boxes = []
        for det in raw_dets:
            boxes.append(DetectionBox(
                xyxy=[det[0], det[1], det[2], det[3]],
                conf=det[4],
                cls_id=det[5],
                track_id=None
            ))
        return [DetectionResult(boxes=boxes, orig_shape=frame.shape[:2])]

    predict = __call__
