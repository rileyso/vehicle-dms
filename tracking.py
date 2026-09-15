"""Core tracking logic for the driver-monitoring demo.

Pure functions + three small state holders (blink, head pose, hysteresis label).
No I/O or display here; main.py does capture and overlay drawing.

All thresholds are project proposals, not physiological/validated values.
EAR is computed in PIXEL coordinates (see research notes): normalized
landmarks are multiplied by image width/height first, so a non-square image
does not distort the ratio.
"""

import math

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe Face Landmarker (478 landmarks) indices.
#
# Six-point eye sets, ordered [corner, upper, upper, corner, lower, lower].
# EAR = (|p2-p6| + |p3-p5|) / (2|p1-p4|)  with p1,p4 the corners.
# ---------------------------------------------------------------------------
EYE_L = [362, 385, 387, 263, 373, 380]  # subject's left eye (image-right)
EYE_R = [33, 160, 158, 133, 153, 144]   # subject's right eye (image-left)

# Iris landmarks: 5 per eye, first is the center. Grouped by image side to
# match the eye constants above (verify left/right visually; the pairing only
# matters for the unused gaze stub).
IRIS_L = [473, 474, 475, 476, 477]
IRIS_R = [468, 469, 470, 471, 472]

# solvePnP correspondences: landmark index -> generic symmetric 3D face model.
# Units are arbitrary (cm-ish); only the *shape* matters because orientation is
# reported relative to a calibrated neutral pose. x: +image-right, y: +up,
# z: +toward camera (nose tip closest).
FACE_MODEL = {
    1:   (0.0, 0.0, 0.0),     # nose tip
    152: (0.0, -7.5, -1.5),   # chin
    33:  (-3.0, 2.5, -3.0),   # right eye outer corner (image-left)
    263: (3.0, 2.5, -3.0),    # left eye outer corner (image-right)
    61:  (3.5, -4.0, -2.5),   # left mouth corner (image-right)
    291: (-3.5, -4.0, -2.5),  # right mouth corner (image-left)
}
MODEL_INDICES = sorted(FACE_MODEL)
MODEL_PTS = np.array([FACE_MODEL[i] for i in MODEL_INDICES], dtype=np.float64)


def landmarks_to_pixels(landmarks, w, h):
    """Convert a list of NormalizedLandmark to a (N,3) pixel array."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts


def ear(points):
    """Eye aspect ratio from 6 pixel points ordered [c,u,u,c,l,l]."""
    p = np.asarray(points, dtype=np.float64)
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    width = 2.0 * np.linalg.norm(p[0] - p[3])
    if width < 1e-6 or not np.isfinite(vertical):
        return None
    return float(vertical / width)


def iris_offset(px, eye_idx, iris_center_idx):
    """Normalized horizontal iris position within an eye, in [-1, 1].

    0 = midway between the eye corners. Uncalibrated; not a gaze estimate.
    """
    a = px[eye_idx[0]][:2]
    b = px[eye_idx[3]][:2]
    c = px[iris_center_idx][:2]
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-9:
        return None
    t = float(np.dot(c - a, ab) / denom)  # 0 at corner a, 1 at corner b
    return 2.0 * t - 1.0


def euler_from_matrix(R):
    """Decompose a rotation matrix to (pitch, yaw, roll) in radians.

    Standard ZYX-style extraction: pitch about x, yaw about y, roll about z.
    See test_euler_from_matrix for the sign/axis validation.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy < 1e-6:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0
    else:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    return pitch, yaw, roll


class BlinkTracker:
    """Timestamp-based bilateral blink state machine with hysteresis.

    An eye is 'closed' below close_threshold and 'open' above reopen_threshold;
    between them it holds its previous state (hysteresis, so the reopen
    threshold must be higher than the close threshold).
    A bilateral event counts once on reopening if the closure lasted
    [min_ms, max_ms]; longer closures become a separate prolonged count.
    """

    def __init__(self, close=0.20, reopen=0.24, min_ms=60, max_ms=500):
        if reopen <= close:
            raise ValueError("reopen threshold must exceed close threshold")
        self.close = close
        self.reopen = reopen
        self.min_ms = min_ms
        self.max_ms = max_ms
        self._left = None
        self._right = None
        self._closed_since = None
        self.blinks = 0
        self.prolonged = 0
        self.total_closed_ms = 0.0

    def _step(self, prev, ear):
        if ear is None:
            return None
        if ear < self.close:
            return True
        if ear > self.reopen:
            return False
        return prev if prev is not None else False

    def update(self, ear_left, ear_right, now_ms):
        self._left = self._step(self._left, ear_left)
        self._right = self._step(self._right, ear_right)
        both_closed = self._left is True and self._right is True
        if both_closed:
            if self._closed_since is None:
                self._closed_since = now_ms
        elif self._closed_since is not None:
            dur = now_ms - self._closed_since
            self.total_closed_ms += dur
            if self.min_ms <= dur <= self.max_ms:
                self.blinks += 1
            elif dur > self.max_ms:
                self.prolonged += 1
            self._closed_since = None

    def reset(self):
        """Called on tracking loss or a frame gap: drop pending events."""
        self._left = None
        self._right = None
        self._closed_since = None

    def is_closed(self):
        return self._left is True and self._right is True

    def time_below_threshold(self, now_ms):
        t = self.total_closed_ms
        if self._closed_since is not None:
            t += now_ms - self._closed_since
        return t


class HeadPoseTracker:
    """Approximate head orientation via OpenCV solvePnP.

    Uses an assumed focal length (f = image width), centred principal point and
    zero distortion -- clearly labelled approximation, fine for a RELATIVE
    orientation demo against a calibrated neutral pose, not for degree-accurate
    absolute pose.
    """

    def __init__(self, w, h):
        self.w, self.h = w, h
        f = float(w)
        self.K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]],
                          dtype=np.float64)
        self.dist = np.zeros((4, 1), dtype=np.float64)
        self.R_neutral = None

    def calibrate(self, R):
        self.R_neutral = R.copy()

    def clear_calibration(self):
        self.R_neutral = None

    def estimate(self, px):
        """px: (N,3) pixel landmark array. Returns a dict or None if invalid."""
        try:
            img_pts = np.array([px[i][:2] for i in MODEL_INDICES],
                               dtype=np.float64)
        except IndexError:
            return None
        if img_pts.shape[0] < 6 or not np.all(np.isfinite(img_pts)):
            return None
        ok, rvec, tvec = cv2.solvePnP(MODEL_PTS, img_pts, self.K, self.dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        rvec = np.asarray(rvec, dtype=np.float64).ravel()
        tvec = np.asarray(tvec, dtype=np.float64).ravel()
        if not (np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))):
            return None
        R, _ = cv2.Rodrigues(rvec)
        proj, _ = cv2.projectPoints(MODEL_PTS, rvec, tvec, self.K, self.dist)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts,
                                           axis=1)))
        euler_abs = euler_from_matrix(R)
        euler_rel = None
        if self.R_neutral is not None:
            euler_rel = euler_from_matrix(R @ self.R_neutral.T)
        return {"R": R, "euler_abs": euler_abs, "euler_rel": euler_rel,
                "reproj_err": err}


class HysteresisLabel:
    """Three-way label with enter/exit hysteresis and a persistence debounce.

    e.g. yaw: labels ("left","neutral","right") with enter=15, exit=10:
    a value beyond +-enter selects the extreme, inside +-exit selects neutral,
    and the reported label only changes after persist_ms of a stable candidate.
    """

    def __init__(self, enter, exit_, persist_ms, labels=("low", "neutral", "high")):
        self.enter = enter
        self.exit = exit_
        self.persist_ms = persist_ms
        self.low, self.neutral, self.high = labels
        self._cand = self.neutral
        self._since = None
        self.value = self.neutral

    def update(self, v, now_ms):
        if v > self.enter:
            cand = self.high
        elif v < -self.enter:
            cand = self.low
        elif abs(v) < self.exit:
            cand = self.neutral
        else:
            cand = self._cand
        if cand != self._cand:
            self._cand = cand
            self._since = now_ms
        if self._since is not None and now_ms - self._since >= self.persist_ms:
            self.value = self._cand
        return self.value

    def reset(self):
        self._cand = self.neutral
        self._since = None
        self.value = self.neutral
