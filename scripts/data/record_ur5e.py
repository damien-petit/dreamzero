#!/usr/bin/env python3
"""
UR5e dataset recorder for DreamZero.

Records robot joint states and camera frames while the robot executes
pre-programmed pendant trajectories. Saves in LeRobot v2.0 format,
ready for convert_lerobot_to_gear.py.

Camera layout (same slot order as DROID):
  left_camera   — USB camera (cv2), external view 1
  right_camera  — ZED2i (pyzed, left image), external view 2
  wrist_camera  — USB camera (cv2), wrist view

Usage:
  python scripts/data/record_ur5e.py \\
      --robot-ip 192.168.1.100 \\
      --output-dir ./data/ur5e_dataset \\
      --left-camera-id 0 \\
      --wrist-camera-id 2 \\
      --fps 15 \\
      --task "pick and place red cube"

After recording, convert to GEAR format:
  python scripts/data/convert_lerobot_to_gear.py \\
      --dataset-path ./data/ur5e_dataset \\
      --embodiment-tag ur5e \\
      --state-keys '{"joint_position": [0, 6], "gripper_position": [6, 7]}' \\
      --action-keys  '{"joint_position": [0, 6], "gripper_position": [6, 7]}' \\
      --relative-action-keys joint_position \\
      --task-key annotation.task
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    import rtde_receive
    HAS_RTDE = True
except ImportError:
    HAS_RTDE = False

try:
    import pyzed.sl as sl
    HAS_ZED = True
except ImportError:
    HAS_ZED = False

try:
    from pyRobotiqGripper import RobotiqGripper
    HAS_ROBOTIQ = True
except ImportError:
    HAS_ROBOTIQ = False


CAMERA_NAMES = ["left_camera", "right_camera", "wrist_camera"]
STATE_DIM = 7   # 6 joints + 1 gripper
ACTION_DIM = 7
CHUNKS_SIZE = 1000


# ---------------------------------------------------------------------------
# Mock hardware (--mock mode)
# ---------------------------------------------------------------------------

class MockRTDE:
    """Simulates UR5e joint positions with a slow sine wave."""
    def __init__(self):
        self._t0 = time.time()

    def getActualQ(self):
        t = time.time() - self._t0
        return [0.1 * np.sin(t + i * 0.5) for i in range(6)]


class MockCamera:
    def __init__(self, height: int = 480, width: int = 640, color: tuple = (80, 80, 80)):
        self.height = height
        self.width = width
        self._color = color
        self._t0 = time.time()

    def read(self) -> np.ndarray:
        frame = np.full((self.height, self.width, 3), self._color, dtype=np.uint8)
        # Animate a moving dot so frames are visually distinct
        t = time.time() - self._t0
        cx = int((np.sin(t) * 0.4 + 0.5) * self.width)
        cy = int((np.cos(t * 0.7) * 0.4 + 0.5) * self.height)
        cv2.circle(frame, (cx, cy), 20, (255, 255, 255), -1)
        return frame

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Camera backends
# ---------------------------------------------------------------------------

class CV2Camera:
    def __init__(self, device_id: int, fps: int):
        self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {device_id}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read USB camera frame")
        return frame  # BGR

    def close(self):
        self.cap.release()


class ZedCamera:
    def __init__(self, resolution_str: str, fps: int):
        if not HAS_ZED:
            raise RuntimeError(
                "pyzed not installed. Install from https://www.stereolabs.com/developers/release/"
            )
        res_map = {
            "VGA": sl.RESOLUTION.VGA,
            "HD720": sl.RESOLUTION.HD720,
            "HD1080": sl.RESOLUTION.HD1080,
        }
        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = res_map[resolution_str]
        init.camera_fps = fps
        status = self.zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED camera open failed: {status}")
        self._mat = sl.Mat()
        info = self.zed.get_camera_information()
        self.width = info.camera_configuration.resolution.width
        self.height = info.camera_configuration.resolution.height

    def read(self) -> np.ndarray:
        if self.zed.grab(sl.RuntimeParameters()) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED grab failed")
        self.zed.retrieve_image(self._mat, sl.VIEW.LEFT)
        return self._mat.get_data()[:, :, :3]  # BGRA → BGR

    def close(self):
        self.zed.close()


# ---------------------------------------------------------------------------
# LeRobot v2.0 writer helpers
# ---------------------------------------------------------------------------

def _write_video(frames: list, path: Path, fps: int):
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _save_episode(episode_idx: int, output_dir: Path, timestamps: list,
                  joint_positions: list, gripper_positions: list,
                  camera_frames: dict, task: str, fps: int):
    T = len(timestamps)
    chunk = episode_idx // CHUNKS_SIZE

    # State: [T, 7]
    joints = np.array(joint_positions, dtype=np.float64)   # [T, 6]
    grippers = np.array(gripper_positions, dtype=np.float64)[:, None]  # [T, 1]
    state_arr = np.concatenate([joints, grippers], axis=1)

    # Action = next-step state (repeat last row)
    action_arr = np.empty_like(state_arr)
    action_arr[:-1] = state_arr[1:]
    action_arr[-1] = state_arr[-1]

    t0 = timestamps[0]
    df = pd.DataFrame({
        "observation.state": list(state_arr),
        "action": list(action_arr),
        "timestamp": [t - t0 for t in timestamps],
        "frame_index": list(range(T)),
        "annotation.task": [task] * T,
    })

    parquet_path = (
        output_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_idx:06d}.parquet"
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    for cam_name, frames in camera_frames.items():
        video_key = f"observation.images.{cam_name}"
        video_path = (
            output_dir / "videos" / f"chunk-{chunk:03d}"
            / video_key / f"episode_{episode_idx:06d}.mp4"
        )
        _write_video(frames, video_path, fps)

    print(f"  Saved episode {episode_idx}: {T} frames  →  {parquet_path.parent}")


def _write_info_json(output_dir: Path, fps: int, total_episodes: int, cam_shapes: dict):
    features = {
        "observation.state": {
            "dtype": "float64",
            "shape": [STATE_DIM],
            "names": ["j0", "j1", "j2", "j3", "j4", "j5", "gripper"],
        },
        "action": {
            "dtype": "float64",
            "shape": [ACTION_DIM],
            "names": ["j0", "j1", "j2", "j3", "j4", "j5", "gripper"],
        },
        "timestamp": {"dtype": "float64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "annotation.task": {"dtype": "string", "shape": [1]},
    }
    for cam_name, (h, w) in cam_shapes.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.fps": fps, "video.channels": 3},
        }
    info = {
        "total_episodes": total_episodes,
        "fps": fps,
        "chunks_size": CHUNKS_SIZE,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    info_path = output_dir / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class Recorder:
    def __init__(self, args):
        self.fps = args.fps
        self.output_dir = Path(args.output_dir)
        self.task = args.task
        self.episode_idx = args.episode_start
        self._stop = threading.Event()
        self._lock = threading.Lock()

        if args.mock:
            print("*** MOCK MODE — no real hardware used ***")
            self.rtde = MockRTDE()
            self.gripper = None
            self.cam_left  = MockCamera(480, 640, (60, 80, 60))
            self.cam_right = MockCamera(720, 1280, (60, 60, 100))
            self.cam_wrist = MockCamera(480, 640, (100, 60, 60))
        else:
            if not HAS_RTDE:
                print("ERROR: ur_rtde not installed. Run: pip install ur_rtde")
                sys.exit(1)
            print(f"Connecting to UR5e at {args.robot_ip} ...")
            self.rtde = rtde_receive.RTDEReceiveInterface(args.robot_ip)  # type: ignore[attr-defined]
            print("  Robot connected.")

            self.gripper = None
            if args.gripper_port:
                if HAS_ROBOTIQ:
                    try:
                        self.gripper = RobotiqGripper()
                        self.gripper.connect(args.gripper_port, 115200)
                        print(f"  Robotiq gripper connected on {args.gripper_port}.")
                    except Exception as e:
                        print(f"  WARNING: Gripper connect failed ({e}). Using 0.0.")
                else:
                    print("  WARNING: pyRobotiqGripper not installed. Using 0.0.")

            print("Opening cameras ...")
            self.cam_left  = CV2Camera(args.left_camera_id, args.fps)
            self.cam_right = ZedCamera(args.zed_resolution, args.fps)
            self.cam_wrist = CV2Camera(args.wrist_camera_id, args.fps)

        self.cam_shapes = {
            "left_camera":  (self.cam_left.height,  self.cam_left.width),
            "right_camera": (self.cam_right.height, self.cam_right.width),
            "wrist_camera": (self.cam_wrist.height, self.cam_wrist.width),
        }
        src = "(mock)" if args.mock else "(USB / ZED2i / USB)"
        print(f"  left_camera  {src}: {self.cam_shapes['left_camera']}")
        print(f"  right_camera {src}: {self.cam_shapes['right_camera']}")
        print(f"  wrist_camera {src}: {self.cam_shapes['wrist_camera']}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _read_gripper(self) -> float:
        if self.gripper is None:
            return 0.0
        try:
            return float(self.gripper.get_current_position()) / 255.0
        except Exception:
            return 0.0

    def _poll_loop(self, timestamps, joint_positions, gripper_positions, camera_frames):
        interval = 1.0 / self.fps
        deadline = time.monotonic()
        while not self._stop.is_set():
            t = time.time()
            joints = np.array(self.rtde.getActualQ(), dtype=np.float64)
            gripper = self._read_gripper()
            left = self.cam_left.read()
            right = self.cam_right.read()
            wrist = self.cam_wrist.read()

            with self._lock:
                timestamps.append(t)
                joint_positions.append(joints)
                gripper_positions.append(gripper)
                camera_frames["left_camera"].append(left)
                camera_frames["right_camera"].append(right)
                camera_frames["wrist_camera"].append(wrist)

            deadline += interval
            sleep_for = deadline - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _record_one(self) -> bool:
        timestamps, joint_positions, gripper_positions = [], [], []
        camera_frames = {k: [] for k in CAMERA_NAMES}

        self._stop.clear()
        t = threading.Thread(
            target=self._poll_loop,
            args=(timestamps, joint_positions, gripper_positions, camera_frames),
            daemon=True,
        )
        t.start()

        input("  [RECORDING] Press Enter to stop ...")
        self._stop.set()
        t.join()

        T = len(timestamps)
        print(f"  Stopped — {T} frames ({T / self.fps:.1f} s)")

        if T < 5:
            print("  Episode too short (< 5 frames), discarding.")
            return False

        _save_episode(
            self.episode_idx, self.output_dir,
            timestamps, joint_positions, gripper_positions,
            camera_frames, self.task, self.fps,
        )
        _write_info_json(self.output_dir, self.fps, self.episode_idx + 1, self.cam_shapes)
        self.episode_idx += 1
        return True

    def run(self):
        print(f"\nTask: '{self.task}'  |  FPS: {self.fps}  |  Output: {self.output_dir}")
        print("Press Enter to START an episode, then Enter again to STOP.")
        print("Type 'x' + Enter to quit.\n")
        try:
            while True:
                cmd = input(f"[Episode {self.episode_idx}] Enter to START (x=quit): ").strip().lower()
                if cmd == "x":
                    break
                self._record_one()
        finally:
            self._close()
        print(f"\nDone. {self.episode_idx} total episodes in {self.output_dir}")
        if self.episode_idx > 0:
            print("\nNext — convert to GEAR format:")
            print(f"  python scripts/data/convert_lerobot_to_gear.py \\")
            print(f"      --dataset-path {self.output_dir} \\")
            print(f"      --embodiment-tag ur5e \\")
            print(f"      --state-keys '{{\"joint_position\": [0, 6], \"gripper_position\": [6, 7]}}' \\")
            print(f"      --action-keys '{{\"joint_position\": [0, 6], \"gripper_position\": [6, 7]}}' \\")
            print(f"      --relative-action-keys joint_position \\")
            print(f"      --task-key annotation.task")

    def _close(self):
        self.cam_left.close()
        self.cam_right.close()
        self.cam_wrist.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="UR5e dataset recorder for DreamZero (LeRobot v2.0 format)"
    )
    p.add_argument("--robot-ip", default="100.80.196.7", help="UR5e controller IP address (required unless --mock)")
    p.add_argument("--output-dir", required=True, help="Output dataset directory")
    p.add_argument("--left-camera-id", type=int, default=0,
                   help="cv2 device index for left_camera (USB, external view 1)")
    p.add_argument("--wrist-camera-id", type=int, default=2,
                   help="cv2 device index for wrist_camera (USB)")
    p.add_argument("--fps", type=int, default=15, help="Recording frequency in Hz")
    p.add_argument("--task", default="robot task",
                   help="Task description string stored in annotation.task")
    p.add_argument("--zed-resolution", default="HD720", choices=["VGA", "HD720", "HD1080"],
                   help="ZED2i capture resolution (right_camera)")
    p.add_argument("--gripper-port", default=None,
                   help="Robotiq gripper serial port, e.g. /dev/ttyUSB0 (optional)")
    p.add_argument("--episode-start", type=int, default=0,
                   help="Starting episode index — use to resume a previous recording session")
    p.add_argument("--mock", action="store_true",
                   help="Mock mode: use synthetic robot state and camera frames (no hardware needed)")
    args = p.parse_args()

    Recorder(args).run()


if __name__ == "__main__":
    main()
