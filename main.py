"""Driver-monitoring demo: head orientation + blink + iris overlay.

Run:  python main.py                 # webcam
      python main.py --video <path>  # replay a video file (e.g. DMD clip)

Controls: q quit, c calibrate neutral head pose, r reset counters.
"""

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from tracking import (
    BlinkTracker,
    HeadPoseTracker,
    HysteresisLabel,
    EYE_L,
    EYE_R,
    IRIS_L,
    IRIS_R,
    landmarks_to_pixels,
    ear,
    iris_offset,
)

MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"

GAP_MS = 100.0           # frame gap above this = interrupted observation
YAW = ("right", "neutral", "left")
PITCH = ("down", "neutral", "up")
ENTER_DEG, EXIT_DEG, PERSIST_MS = 15.0, 10.0, 300


def make_landmarker(model_path):
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def put(frame, text, y, color=(0, 255, 0)):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
                cv2.LINE_AA)


def draw_eye(frame, pts6, color):
    poly = pts6[:, :2].astype(int)
    cv2.polylines(frame, [poly], True, color, 1, cv2.LINE_AA)
    for x, y in poly:
        cv2.circle(frame, (x, y), 2, color, -1, cv2.LINE_AA)


def draw_iris(frame, px, iris_idx):
    for i in iris_idx[1:]:
        x, y = px[i][:2].astype(int)
        cv2.circle(frame, (x, y), 2, (255, 255, 0), -1, cv2.LINE_AA)
    x, y = px[iris_idx[0]][:2].astype(int)
    cv2.circle(frame, (x, y), 4, (0, 255, 255), -1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", help="video file to replay instead of the webcam")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--mirror", action="store_true", default=None,
                   help="mirror display (default: on for webcam, off for video)")
    p.add_argument("--close", type=float, default=0.20, help="EAR close threshold")
    p.add_argument("--reopen", type=float, default=0.24, help="EAR reopen threshold")
    p.add_argument("--auto-calibrate", action="store_true",
                   help="calibrate neutral pose on the first valid frame")
    args = p.parse_args()

    if args.reopen <= args.close:
        raise SystemExit("--reopen must be greater than --close (hysteresis)")

    cap = cv2.VideoCapture(args.video) if args.video else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("could not open camera/video")

    landmarker = make_landmarker(MODEL_PATH)
    blink = BlinkTracker(close=args.close, reopen=args.reopen)
    yaw_label = HysteresisLabel(ENTER_DEG, EXIT_DEG, PERSIST_MS, labels=YAW)
    pitch_label = HysteresisLabel(ENTER_DEG, EXIT_DEG, PERSIST_MS, labels=PITCH)
    mirror = args.mirror if args.mirror is not None else (args.video is None)

    head = None
    last_ms = None
    fps, win_start, win_frames = 0.0, None, 0

    print("controls: q=quit  c=calibrate neutral  r=reset counters")
    if args.video:
        print(f"replaying: {args.video}")

    while True:
        ok, frame = cap.read()
        now_ms = time.monotonic() * 1000.0

        if not ok or frame is None:
            if args.video:
                break
            cv2.waitKey(1)
            continue

        if mirror:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        if head is None or head.w != w or head.h != h:
            head = HeadPoseTracker(w, h)

        if last_ms is not None and (now_ms - last_ms) > GAP_MS:
            blink.reset()
            yaw_label.reset()
            pitch_label.reset()
        last_ms = now_ms

        # --- detect ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=np.ascontiguousarray(rgb))
        result = landmarker.detect_for_video(mp_img, int(now_ms))

        face_ok = bool(result.face_landmarks)
        px = None
        if face_ok and len(result.face_landmarks[0]) >= 478:
            px = landmarks_to_pixels(result.face_landmarks[0], w, h)

        # --- blink (bilateral; both eyes must be usable) ---
        ear_l = ear(px[EYE_L]) if px is not None else None
        ear_r = ear(px[EYE_R]) if px is not None else None
        if ear_l is None or ear_r is None:
            blink.reset()
        else:
            blink.update(ear_l, ear_r, now_ms)

        # --- head pose ---
        pose = head.estimate(px) if px is not None else None

        head_txt, yaw_deg, pitch_deg, roll_deg = "unknown", None, None, None
        if pose is not None:
            e = pose["euler_rel"] if pose["euler_rel"] is not None else pose["euler_abs"]
            pitch_deg, yaw_deg, roll_deg = (math.degrees(x) for x in e)
            if pose["euler_rel"] is not None:
                head_txt = (f"{yaw_label.update(yaw_deg, now_ms)} / "
                            f"{pitch_label.update(pitch_deg, now_ms)}")
            else:
                yaw_label.reset()
                pitch_label.reset()
                head_txt = "uncalibrated"

        # --- controls ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c") and pose is not None:
            head.calibrate(pose["R"])
            print("neutral head pose calibrated")
        if key == ord("r"):
            blink = BlinkTracker(close=args.close, reopen=args.reopen)
            yaw_label.reset()
            pitch_label.reset()
            head.clear_calibration()
            print("counters reset")

        if args.auto_calibrate and head.R_neutral is None and pose is not None:
            head.calibrate(pose["R"])

        # --- overlay ---
        if px is not None:
            draw_eye(frame, px[EYE_L], (0, 255, 0))
            draw_eye(frame, px[EYE_R], (0, 255, 0))
            draw_iris(frame, px, IRIS_L)
            draw_iris(frame, px, IRIS_R)

        if not face_ok:
            status, color = "NO FACE", (0, 0, 255)
        elif ear_l is None or ear_r is None:
            status, color = "EYES UNKNOWN", (0, 200, 255)
        else:
            status, color = "tracking", (0, 255, 0)

        # rolling fps over a 1 s window
        if win_start is None:
            win_start = now_ms
        win_frames += 1
        if now_ms - win_start >= 1000.0:
            fps = win_frames * 1000.0 / (now_ms - win_start)
            win_start, win_frames = now_ms, 0

        put(frame, f"status: {status}", 20, color)
        put(frame, f"fps: {fps:.0f}", 40)
        put(frame, f"head: {head_txt}", 60)
        if yaw_deg is not None:
            put(frame, f"  yaw {yaw_deg:+.0f}  pitch {pitch_deg:+.0f}  roll {roll_deg:+.0f}", 80)
        put(frame, f"blinks: {blink.blinks}   prolonged: {blink.prolonged}", 100)
        put(frame, f"time below EAR threshold: {blink.time_below_threshold(now_ms):.1f} s", 120)
        if ear_l is not None and ear_r is not None:
            put(frame, f"EAR L {ear_l:.2f}  R {ear_r:.2f}", 140)
            io = iris_offset(px, EYE_L, IRIS_L[0])
            if io is not None:
                put(frame, f"iris offset (uncalibrated): {io:+.2f}", 160)
        if not face_ok or ear_l is None or ear_r is None:
            put(frame, "calibrate head pose with 'c'", h - 10, (0, 200, 255))

        cv2.imshow("driver monitoring", frame)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
