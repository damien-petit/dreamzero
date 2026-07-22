# UR5e Data Capture on the RTX 5090 Machine — Cameras and Recording

How to check the camera views and record UR5e demonstrations on the lab RTX 5090
(x86_64) machine. For the follow-up steps — GEAR conversion and fine-tuning — see
[UR5E_DATA_COLLECTION_AND_FINETUNING.md](UR5E_DATA_COLLECTION_AND_FINETUNING.md)
(its ZED ARM64 install section applies to the GX10, not this machine).

---

## Hardware Setup

| Component | Details |
|---|---|
| Robot | UR5e at `100.80.196.7`, RTDE enabled on the pendant |
| Gripper | Robotiq, wired to the **tool connector**, controlled by the Robotiq URCap — read over the URCap socket server (`<robot-ip>:63352`), no USB cable to the PC |
| `left_camera` | Innomaker U20CAM (USB), external view 1 |
| `right_camera` | ZED 2i (via pyzed SDK), external view 2 |
| `wrist_camera` | Innomaker U20CAM (USB), wrist view |

**Enable RTDE on the pendant:** Settings → System → Real-Time Data Exchange → ON

**Activate the environment** (every new terminal):
```bash
source /home/d/miniconda3/bin/activate dreamzero
cd /home/d/devel-src/dreamzero
```

---

## Camera Identification (reboot-proof)

The two Innomaker USB cameras report the **same serial number** (`SN0001`), so
they cannot be told apart by ID, and bare cv2 indices (`/dev/video0`, `4`, …)
shuffle across reboots. Both scripts therefore identify the USB cameras by the
**physical USB port** via `/dev/v4l/by-path/`, which is stable across reboots:

| Camera | Default device |
|---|---|
| `left_camera` | `/dev/v4l/by-path/pci-0000:80:14.0-usb-0:5:1.0-video-index0` |
| `wrist_camera` | `/dev/v4l/by-path/pci-0000:80:14.0-usb-0:8:1.0-video-index0` |

These are the defaults for `--left-camera-id` / `--wrist-camera-id` in both
`record_ur5e.py` and `view_cameras.py` (defined once as `DEFAULT_LEFT_CAMERA` /
`DEFAULT_WRIST_CAMERA` in `scripts/data/record_ur5e.py`). The ZED is opened
through its SDK and needs no device path.

> **Rule:** identity follows the USB **port**, not the camera. Do not swap the
> two USB plugs — label the cables. If a camera stops opening, re-check with:
> ```bash
> v4l2-ctl --list-devices
> ls -l /dev/v4l/by-path/
> ```

---

## Live Camera Viewer

Check framing, focus and camera assignment before recording:

```bash
python3 scripts/data/view_cameras.py
```

Shows all three streams side by side in one window, each tile captioned with
its camera name (`left_camera` / `right_camera` / `wrist_camera`). The viewer
uses the same camera backends and defaults as the recorder, so what you see is
exactly what gets recorded.

| Key | Action |
|---|---|
| `q` or `ESC` | Quit |

Options: `--fps` (default 15), `--zed-resolution VGA|HD720|HD1080` (default
HD720), `--left-camera-id` / `--wrist-camera-id` (index or device path),
`--mock` (synthetic frames, no hardware).

---

## Recording Demonstrations

Each episode = one pendant program execution, recorded start to finish.

```bash
python3 scripts/data/record_ur5e.py \
    --output-dir ./data/ur5e_dataset \
    --gripper-urcap \
    --task "describe the task here"
```

- `--task` is stored on every frame and used as the language instruction during
  training — write a real description.
- `--gripper-urcap` reads the Robotiq gripper through the URCap socket server on
  the robot controller. Without it the gripper channel records a constant `0.0`.
- Robot IP, cameras, ZED resolution and fps (15) all default correctly for this
  machine.

| Control | Action |
|---|---|
| `Enter` | Start recording an episode |
| `Enter` again | Stop and save the episode |
| `x` + `Enter` | Quit |

**Workflow per episode:**
1. Reset the robot to the start position on the pendant
2. Start the pendant program
3. Press `Enter` in the recorder to begin capturing
4. Press `Enter` again when the program finishes

Aim for **at least 50 episodes** per task. Episodes shorter than 5 frames are
discarded automatically.

**Resume a previous session** (avoids overwriting existing episodes):
```bash
python3 scripts/data/record_ur5e.py \
    --output-dir ./data/ur5e_dataset \
    --gripper-urcap \
    --task "describe the task here" \
    --episode-start <N>    # N = number of episodes already saved
```

**Test without hardware:**
```bash
python3 scripts/data/record_ur5e.py --mock --output-dir ./data/ur5e_test --task "test"
```

### Gripper values

The recorded gripper channel is the **actual position** normalized to `[0, 1]`:
`0.0` = fully open, `1.0` = fully closed (Robotiq raw scale 0–255). Because it
is the measured position, a grasp that stops on an object records the true
stopped position.

### Output structure (LeRobot v2.0)

```
data/ur5e_dataset/
├── data/chunk-000/
│   └── episode_000000.parquet         # observation.state / action: [j0..j5, gripper]
├── videos/chunk-000/
│   ├── observation.images.left_camera/episode_000000.mp4
│   ├── observation.images.right_camera/episode_000000.mp4
│   └── observation.images.wrist_camera/episode_000000.mp4
└── meta/
    └── info.json
```

### Verify a recording

```bash
# State/action data — expect state of 7 values [j0..j5, gripper]
python3 -c "
import pandas as pd
df = pd.read_parquet('data/ur5e_dataset/data/chunk-000/episode_000000.parquet')
print(df.shape, df.columns.tolist())
print('State sample:', df['observation.state'].iloc[0])
"

# Videos — check all three actually show motion
ffplay data/ur5e_dataset/videos/chunk-000/observation.images.right_camera/episode_000000.mp4
```

Next step: GEAR conversion and training — see
[UR5E_DATA_COLLECTION_AND_FINETUNING.md](UR5E_DATA_COLLECTION_AND_FINETUNING.md)
(Step 2 onward).

---

## Implementation Notes / Troubleshooting

**`right_camera` video is static** — fixed: pyzed's `Mat.get_data()` returns a
*view* into a reused buffer; `ZedCamera.read()` now copies each frame. If you
still have episodes recorded before this fix, their ZED videos are frozen —
re-record them.

**USB bandwidth** — `CV2Camera` requests MJPG from the USB cameras. Raw YUYV
from two cameras plus the ZED can starve the USB bus and cause intermittent
`Failed to read USB camera frame`.

**ZED depth disabled** — the recorder only needs RGB, so the ZED opens with
`DEPTH_MODE.NONE`. The startup warning `Self-calibration skipped` is expected
and harmless.

**`Cannot open camera device …`** — the camera is unplugged, in use by another
process (close the viewer before recording), or plugged into a different USB
port than its by-path default. Check `v4l2-ctl --list-devices`.

**Gripper reads `0.0` constantly** — you forgot `--gripper-urcap`, or the URCap
socket server is unreachable. Quick test:
```bash
python3 -c "
import sys; sys.path.insert(0, 'scripts/data')
from record_ur5e import URCapGripper
g = URCapGripper('100.80.196.7'); print('position:', g.get_current_position()); g.close()
"
```

**RTDE connection refused** — enable RTDE on the pendant (Settings → System →
Real-Time Data Exchange) and confirm the robot is reachable:
`ping 100.80.196.7`.
