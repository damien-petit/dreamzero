#!/usr/bin/env python3
"""
Live viewer for the three UR5e recording cameras.

Shows left_camera (USB), right_camera (ZED2i) and wrist_camera (USB) side by
side in one window, each tile captioned with its name. Use it to check camera
indices, framing and focus before recording with record_ur5e.py.

Usage:
  python scripts/data/view_cameras.py --left-camera-id 0 --wrist-camera-id 2

  q or ESC — quit

Test without hardware:
  python scripts/data/view_cameras.py --mock
"""

import argparse
import time

import cv2
import numpy as np

from record_ur5e import (
    DEFAULT_LEFT_CAMERA,
    DEFAULT_WRIST_CAMERA,
    CV2Camera,
    MockCamera,
    ZedCamera,
    _camera_device,
)

TILE_HEIGHT = 360
WINDOW_NAME = "UR5e cameras (q/ESC to quit)"


def make_tile(frame: np.ndarray, name: str, height: int = TILE_HEIGHT) -> np.ndarray:
    """Resize a frame to a common height and draw a caption banner on it."""
    h, w = frame.shape[:2]
    scale = height / h
    tile = cv2.resize(frame, (int(round(w * scale)), height))
    tile = np.ascontiguousarray(tile)

    (tw, th), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    banner_h = th + baseline + 12
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, 0), (tw + 20, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, tile, 0.4, 0, tile)
    cv2.putText(tile, name, (10, th + 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return tile


def compose(frames: dict) -> np.ndarray:
    """Stack the named frames horizontally into one canvas."""
    return np.hstack([make_tile(frame, name) for name, frame in frames.items()])


def main():
    p = argparse.ArgumentParser(description="Live side-by-side view of the three UR5e cameras")
    p.add_argument("--left-camera-id", type=_camera_device, default=DEFAULT_LEFT_CAMERA,
                   help="left_camera (USB, external view 1): cv2 index or stable "
                        "device path, e.g. /dev/v4l/by-path/...-video-index0")
    p.add_argument("--wrist-camera-id", type=_camera_device, default=DEFAULT_WRIST_CAMERA,
                   help="wrist_camera (USB): cv2 index or stable device path")
    p.add_argument("--fps", type=int, default=15, help="Capture frequency in Hz")
    p.add_argument("--zed-resolution", default="HD720", choices=["VGA", "HD720", "HD1080"],
                   help="ZED2i capture resolution (right_camera)")
    p.add_argument("--mock", action="store_true",
                   help="Mock mode: synthetic frames, no hardware needed")
    args = p.parse_args()

    if args.mock:
        print("*** MOCK MODE — no real hardware used ***")
        cameras = {
            "left_camera":  MockCamera(480, 640, (60, 80, 60)),
            "right_camera": MockCamera(720, 1280, (60, 60, 100)),
            "wrist_camera": MockCamera(480, 640, (100, 60, 60)),
        }
    else:
        print("Opening cameras ...")
        cameras = {
            "left_camera":  CV2Camera(args.left_camera_id, args.fps),
            "right_camera": ZedCamera(args.zed_resolution, args.fps),
            "wrist_camera": CV2Camera(args.wrist_camera_id, args.fps),
        }
    for name, cam in cameras.items():
        print(f"  {name}: {cam.height}x{cam.width}" if not args.mock
              else f"  {name}: mock")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    interval = 1.0 / args.fps
    try:
        while True:
            t0 = time.monotonic()
            canvas = compose({name: cam.read() for name, cam in cameras.items()})
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            sleep_for = interval - (time.monotonic() - t0)
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        for cam in cameras.values():
            cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
