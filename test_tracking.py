"""Runnable self-checks for tracking.py. No camera, no model needed.

Run:  .venv/Scripts/python.exe test_tracking.py
Exits non-zero if any check fails.
"""

import math

import cv2
import numpy as np

from tracking import (
    BlinkTracker,
    HeadPoseTracker,
    HysteresisLabel,
    MODEL_INDICES,
    MODEL_PTS,
    ear,
    euler_from_matrix,
    landmarks_to_pixels,
)

PASS = 0


def check(name, fn):
    global PASS
    try:
        fn()
        PASS += 1
        print(f"[ok] {name}")
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")
        raise


def test_ear_geometry():
    # 100 px wide, 30 px open vs 4 px closed -> EAR 0.30 vs 0.04
    open_eye = np.array([[0, 0], [30, -15], [70, -15], [100, 0], [70, 15], [30, 15]], float)
    closed_eye = np.array([[0, 0], [30, -2], [70, -2], [100, 0], [70, 2], [30, 2]], float)
    assert abs(ear(open_eye) - 0.30) < 1e-9, ear(open_eye)
    assert abs(ear(closed_eye) - 0.04) < 1e-9, ear(closed_eye)
    assert ear(open_eye) > ear(closed_eye)


def test_landmarks_to_pixels():
    class LM:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    lm = [LM(0.5, 0.25, 0.0)]
    px = landmarks_to_pixels(lm, 640, 480)
    assert px[0][0] == 320.0 and px[0][1] == 120.0, px[0]


def test_blink_state_machine():
    b = BlinkTracker(close=0.20, reopen=0.24)
    open_, closed = 0.35, 0.05
    b.update(open_, open_, 0.0)                 # open
    b.update(closed, closed, 100.0)             # both close
    b.update(open_, open_, 200.0)               # reopen after 100 ms
    assert b.blinks == 1 and b.prolonged == 0
    assert abs(b.time_below_threshold(300.0) - 100.0) < 1e-6

    b.update(closed, closed, 1000.0)            # long closure
    b.update(open_, open_, 2000.0)              # 1000 ms -> prolonged
    assert b.blinks == 1 and b.prolonged == 1

    # wink (one eye only) must not count
    b.update(closed, open_, 3000.0)
    b.update(open_, open_, 3100.0)
    assert b.blinks == 1 and b.prolonged == 1

    # reset drops a pending closure
    b.update(closed, closed, 4000.0)
    b.reset()
    b.update(open_, open_, 4100.0)
    assert b.blinks == 1 and b.prolonged == 1

    # hysteresis: between thresholds holds previous state
    h = BlinkTracker(close=0.20, reopen=0.24)
    h.update(0.35, 0.35, 0.0)      # open
    h.update(0.22, 0.22, 100.0)    # in hysteresis band -> stays open
    assert h.is_closed() is False
    h.update(0.05, 0.05, 200.0)    # below close -> closed
    assert h.is_closed() is True
    h.update(0.22, 0.22, 300.0)    # still in band -> stays closed
    assert h.is_closed() is True


def test_hysteresis_label():
    h = HysteresisLabel(enter=15.0, exit_=10.0, persist_ms=300,
                        labels=("left", "neutral", "right"))
    assert h.update(0.0, 0) == "neutral"
    assert h.update(20.0, 100) == "neutral"       # candidate right, not persisted
    assert h.update(20.0, 500) == "right"         # persisted
    assert h.update(12.0, 600) == "right"         # in band -> holds
    assert h.update(0.0, 800) == "right"          # candidate neutral, not persisted
    assert h.update(0.0, 1200) == "neutral"       # persisted
    assert h.update(-20.0, 1500) == "neutral"
    assert h.update(-20.0, 1900) == "left"


def test_euler_from_matrix():
    # rotation about +y by 0.35 rad -> yaw ~= 0.35
    rvec = np.array([0.0, 0.35, 0.0])
    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = euler_from_matrix(R)
    assert abs(yaw - 0.35) < 1e-6, yaw
    assert abs(pitch) < 1e-6 and abs(roll) < 1e-6


def test_head_pose_synthetic():
    w, h = 640, 480
    t = HeadPoseTracker(w, h)
    rvec = np.array([0.0, math.radians(20.0), 0.0])   # yaw 20 deg
    tvec = np.array([0.0, 0.0, 30.0])
    img_pts, _ = cv2.projectPoints(MODEL_PTS, rvec, tvec, t.K, t.dist)
    px = np.zeros((478, 3), dtype=np.float64)
    for k, idx in enumerate(MODEL_INDICES):
        px[idx][:2] = img_pts[k][0]
    res = t.estimate(px)
    assert res is not None
    assert res["reproj_err"] < 1.0, res["reproj_err"]
    yaw_deg = math.degrees(res["euler_abs"][1])
    assert abs(yaw_deg - 20.0) < 3.0, yaw_deg

    # relative to a calibrated neutral = identity
    t.calibrate(np.eye(3))
    assert t.R_neutral is not None


if __name__ == "__main__":
    check("ear geometry", test_ear_geometry)
    check("landmarks_to_pixels", test_landmarks_to_pixels)
    check("blink state machine", test_blink_state_machine)
    check("hysteresis label", test_hysteresis_label)
    check("euler_from_matrix", test_euler_from_matrix)
    check("head pose synthetic", test_head_pose_synthetic)
    print(f"\n{PASS} checks passed")
