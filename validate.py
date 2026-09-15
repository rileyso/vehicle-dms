"""Validate blink/head-pose detection against a DMD OpenLABEL annotation.

Usage:
  python validate.py --video <rgb_face.mp4> --annotation <ann_drowsiness.json>
  python validate.py --video <...> --annotation <...> --thresholds 0.10 0.18 0.20

Reports face/eye availability, EAR open-vs-closed separation, a blink
threshold sweep (precision/recall/F1 against the annotation's
'blinks/blinking' intervals), and a head-pose sanity summary.

DMD-specific: parses OpenLABEL 1.0 'eyes_state/{open,close}' and
'blinks/blinking' action intervals. No display, no camera.
"""

import argparse
import json
import sys

import cv2
import numpy as np
import mediapipe as mp

import main as M
from tracking import EYE_L, EYE_R, HeadPoseTracker, ear, landmarks_to_pixels

DEFAULT_THRESHOLDS = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
BLINK_MIN_MS, BLINK_MAX_MS = 60, 500


def load_gt(ann_path):
    d = json.load(open(ann_path, encoding="utf-8"))
    acts = d["openlabel"]["actions"]

    def intervals(t):
        for a in acts.values():
            if a.get("type") == t:
                return [(v["frame_start"], v["frame_end"])
                        for v in a.get("frame_intervals", [])]
        return []

    return {"open": intervals("eyes_state/open"),
            "close": intervals("eyes_state/close"),
            "blink": intervals("blinks/blinking")}


def in_any(ivs, f):
    return any(s <= f <= e for s, e in ivs)


def process(video, landmarker, max_frames=None):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    el = np.full(total, np.nan)
    er = np.full(total, np.nan)
    face_frames = eye_frames = 0
    yaws, pitches, errs = [], [], []
    head = None
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and f >= max_frames):
            break
        h, w = frame.shape[:2]
        if head is None:
            head = HeadPoseTracker(w, h)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=np.ascontiguousarray(rgb)), int(f * 1000 / fps))
        if res.face_landmarks and len(res.face_landmarks[0]) >= 478:
            face_frames += 1
            px = landmarks_to_pixels(res.face_landmarks[0], w, h)
            el[f], er[f] = ear(px[EYE_L]), ear(px[EYE_R])
            if np.isfinite(el[f]) and np.isfinite(er[f]):
                eye_frames += 1
            pose = head.estimate(px)
            if pose is not None:
                p_, y_, r_ = pose["euler_abs"]
                yaws.append(np.degrees(y_))
                pitches.append(np.degrees(p_))
                errs.append(pose["reproj_err"])
        f += 1
    cap.release()
    return {"fps": fps, "n": f, "el": el[:f], "er": er[:f],
            "face_frames": face_frames, "eye_frames": eye_frames,
            "yaws": yaws, "pitches": pitches, "errs": errs}


def closed_runs(mn, thr, n):
    runs, s = [], None
    for i in range(n):
        if mn[i] < thr and s is None:
            s = i
        if not mn[i] < thr and s is not None:
            runs.append((s, i - 1))
            s = None
    if s is not None:
        runs.append((s, n - 1))
    return runs


def match(runs, gt_blink):
    tp, matched = 0, set()
    for p in runs:
        plen = p[1] - p[0] + 1
        for gi, g in enumerate(gt_blink):
            ov = max(0, min(p[1], g[1]) - max(p[0], g[0]) + 1)
            if ov / plen > 0.5:
                tp += 1
                matched.add(gi)
                break
    return tp, len(matched)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--thresholds", type=float, nargs="*",
                    default=DEFAULT_THRESHOLDS)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    gt = load_gt(args.annotation)
    lm = M.make_landmarker(M.MODEL_PATH)
    r = process(args.video, lm, args.max_frames)
    lm.close()

    n, fps = r["n"], r["fps"]
    mn = np.minimum(r["el"], r["er"])

    print(f"video: {args.video}")
    print(f"frames: {n}  fps: {fps:.2f}")
    print(f"face detected:   {r['face_frames']}  ({100*r['face_frames']/n:.1f}%)")
    print(f"both eyes valid: {r['eye_frames']}  ({100*r['eye_frames']/n:.1f}%)")
    print()

    print("=== EAR distribution (min of both eyes) ===")
    for key, name in [("open", "open"), ("blink", "blink"), ("close", "close")]:
        vals = mn[[in_any(gt[key], i) for i in range(n)]]
        if len(vals):
            print(f"  {name:6s} n={len(vals):5d}  median={np.median(vals):.3f}  "
                  f"p25={np.percentile(vals, 25):.3f}  p75={np.percentile(vals, 75):.3f}")
    print()

    ms = lambda iv: (iv[1] - iv[0] + 1) * 1000 / fps
    gt_blink = gt["blink"]
    gt_close_ms = sum(ms(iv) for iv in gt["close"])
    print(f"GT: {len(gt_blink)} blinks, {len(gt['close'])} close-intervals "
          f"({gt_close_ms/1000:.1f}s close)")
    print()
    print(f"=== threshold sweep (bilateral, {BLINK_MIN_MS}-{BLINK_MAX_MS}ms blink) ===")
    print("thr   blinks  prolong  TP  prec   recall  F1")
    for thr in args.thresholds:
        runs = closed_runs(mn, thr, n)
        blinks = [iv for iv in runs if BLINK_MIN_MS <= ms(iv) <= BLINK_MAX_MS]
        tp, matched = match(blinks, gt_blink)
        prec = tp / len(blinks) if blinks else 0.0
        rec = matched / len(gt_blink) if gt_blink else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"{thr:.2f}  {len(blinks):5d}  {len(runs)-len(blinks):6d}  "
              f"{tp:2d}  {prec:.2f}   {rec:.2f}   {f1:.2f}")

    if r["yaws"]:
        print()
        print("=== head pose (absolute, sanity only) ===")
        print(f"yaw   min/med/max: {min(r['yaws']):+.0f} / "
              f"{np.median(r['yaws']):+.0f} / {max(r['yaws']):+.0f} deg")
        print(f"pitch min/med/max: {min(r['pitches']):+.0f} / "
              f"{np.median(r['pitches']):+.0f} / {max(r['pitches']):+.0f} deg")
        print(f"reproj err med/p95: {np.median(r['errs']):.1f} / "
              f"{np.percentile(r['errs'], 95):.1f} px")


if __name__ == "__main__":
    sys.exit(main())
