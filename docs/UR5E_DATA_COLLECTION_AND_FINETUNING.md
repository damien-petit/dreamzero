# UR5e — Data Collection, GEAR Conversion, and DreamZero Fine-Tuning

Step-by-step guide to record demonstrations with a UR5e robot, convert them to GEAR format, and fine-tune a DreamZero policy on the new embodiment.

> **Recording on the lab RTX 5090 (x86_64) machine?** See
> [UR5E_RTX5090_DATA_CAPTURE.md](UR5E_RTX5090_DATA_CAPTURE.md) for that
> machine's camera setup (stable `/dev/v4l/by-path` devices, live viewer) and
> the URCap-based gripper readout (`--gripper-urcap`).

---

## Prerequisites

| Requirement | Details |
|---|---|
| Robot | UR5e reachable at `100.80.196.7` with RTDE enabled |
| Cameras | USB (`left_camera`, `wrist_camera`) + ZED2i (`right_camera`) |
| ZED SDK | Installed system-wide — see [ZED SDK installation](#zed-sdk-installation-arm64) |
| Python env | `dreamzero` conda environment |

**Enable RTDE on the pendant:**
> Settings → System → Real-Time Data Exchange → ON

**Activate the environment** (every new terminal):
```bash
source /home/d/miniconda3/bin/activate dreamzero
```

**Find camera device indices:**
```bash
ls /dev/video*
```

---

## ZED SDK Installation (ARM64)

The ASUS GX10 is `aarch64` (Grace-Blackwell). The ZED SDK must be installed system-wide before the Python bindings can be installed into the conda env.

**1. Download the installer** from [stereolabs.com/developers/release](https://www.stereolabs.com/developers/release/)
- Platform: Linux
- Architecture: ARM 64-bit (aarch64) — *not* Jetson

**2. Run the installer:**
```bash
chmod +x ZED_SDK_*.run
sudo ./ZED_SDK_*.run -- silent skip_cuda_check
```

> `skip_cuda_check` is required on CUDA 13.0 (the SDK ships for CUDA 12.x but works on 13.0 for image capture).
> If the installer says `Detected Ubuntu24_sbsa, required exact Ubuntu24` — this is expected on Grace hardware. Continue.

**3. Install Python bindings into the dreamzero env:**
```bash
source /home/d/miniconda3/bin/activate dreamzero
python /usr/local/zed/get_python_api.py
```

If it fails due to CUDA version mismatch:
```bash
python /usr/local/zed/get_python_api.py --cuda_version 12.1
```

**Verify:**
```bash
python -c "import pyzed.sl as sl; print(sl.Camera().get_sdk_version())"
```

---

## Step 1 — Record Demonstrations

Each episode = one pendant program execution, recorded from start to finish.

```bash
python3 scripts/data/record_ur5e.py \
    --output-dir ./data/ur5e_dataset \
    --left-camera-id 0 \
    --wrist-camera-id 2 \
    --fps 15 \
    --task "describe the task here"
```

| Control | Action |
|---|---|
| `Enter` | Start recording an episode |
| `Enter` again | Stop and save the episode |
| `x` + `Enter` | Quit |

**Workflow per episode:**
1. Reset the robot to start position on the pendant
2. Start the pendant program
3. Press `Enter` in the recorder to begin capturing
4. Press `Enter` again when the program finishes

Aim for **at least 50 episodes** per task for meaningful training signal.

**Resume a previous session:**
```bash
python3 scripts/data/record_ur5e.py \
    --output-dir ./data/ur5e_dataset \
    --task "describe the task here" \
    --episode-start <N>    # N = number of episodes already saved
```

**Test without hardware (`--mock`):**
```bash
python3 scripts/data/record_ur5e.py --mock \
    --output-dir ./data/ur5e_test --task "test" --fps 15
```

### Output structure

```
data/ur5e_dataset/
├── data/chunk-000/
│   ├── episode_000000.parquet
│   └── ...
├── videos/chunk-000/
│   ├── observation.images.left_camera/episode_000000.mp4
│   ├── observation.images.right_camera/episode_000000.mp4
│   └── observation.images.wrist_camera/episode_000000.mp4
└── meta/
    └── info.json
```

### Verify a recording

```bash
# Parquet
python3 -c "
import pandas as pd
df = pd.read_parquet('data/ur5e_dataset/data/chunk-000/episode_000000.parquet')
print(df.shape, df.columns.tolist())
print('State sample:', df['observation.state'].iloc[0])
"
# Expected: shape (T, 5), state has 7 values [j0..j5, gripper]

# Video
ffplay data/ur5e_dataset/videos/chunk-000/observation.images.left_camera/episode_000000.mp4
```

---

## Step 2 — Convert to GEAR Format

Generates the metadata files DreamZero needs. Does **not** modify parquet files or videos.

```bash
python3 scripts/data/convert_lerobot_to_gear.py \
    --dataset-path ./data/ur5e_dataset \
    --embodiment-tag ur5e \
    --state-keys '{"joint_position": [0, 6], "gripper_position": [6, 7]}' \
    --action-keys  '{"joint_position": [0, 6], "gripper_position": [6, 7]}' \
    --relative-action-keys joint_position \
    --task-key annotation.task
```

**Generated files** (under `data/ur5e_dataset/meta/`):

| File | Contents |
|---|---|
| `modality.json` | Maps state / action / video / annotation keys |
| `embodiment.json` | `{"embodiment_tag": "ur5e"}` |
| `stats.json` | Per-feature statistics (mean, std, min, max, q01, q99) |
| `relative_stats_dreamzero.json` | Relative action statistics (action − reference state) |
| `tasks.jsonl` | Unique task descriptions |
| `episodes.jsonl` | Per-episode metadata (index, length, tasks) |

### Verify conversion

```bash
python3 -c "
import json
m = json.load(open('data/ur5e_dataset/meta/modality.json'))
print('State keys: ', list(m['state'].keys()))
print('Action keys:', list(m['action'].keys()))
print('Video keys: ', list(m['video'].keys()))
"
# Expected video keys: left_camera, right_camera, wrist_camera
# Expected state/action keys: joint_position, gripper_position
```

---

## Step 3 — Fine-Tune DreamZero

### Option A — Wan2.2 5B (recommended — single GPU, faster)

```bash
DATA_ROOT=./data/ur5e_dataset bash scripts/train/ur5e_training.sh
```

| Setting | Value |
|---|---|
| Backbone | Wan2.2-TI2V-5B (`BACKBONE=wan22`) |
| Resolution | 160×320 |
| GPUs | 1 |
| `max_steps` | 5000 |
| Output | `./checkpoints/dreamzero_ur5e_wan22_lora` |

### Option B — Wan2.1 14B (higher capacity — 2+ GPUs)

```bash
BACKBONE=wan21 NUM_GPUS=2 DATA_ROOT=./data/ur5e_dataset bash scripts/train/ur5e_training.sh
```

| Setting | Value |
|---|---|
| Backbone | Wan2.1-I2V-14B (`BACKBONE=wan21`) |
| Resolution | 176×320 |
| Output | `./checkpoints/dreamzero_ur5e_wan21_lora` |

### Useful overrides

```bash
# Quick sanity check (2 steps)
DATA_ROOT=./data/ur5e_dataset max_steps=2 OUTPUT_DIR=/tmp/ur5e_test \
    bash scripts/train/ur5e_training.sh

# Custom output dir
OUTPUT_DIR=./checkpoints/my_run DATA_ROOT=./data/ur5e_dataset \
    bash scripts/train/ur5e_training.sh

# Enable W&B logging
WANDB_MODE=online DATA_ROOT=./data/ur5e_dataset bash scripts/train/ur5e_training.sh
```

Checkpoints are saved every 1000 steps to `OUTPUT_DIR/`.

---

## Step 4 — Verify Training

```bash
# List checkpoints
ls ./checkpoints/dreamzero_ur5e_wan22_lora/

# Run standalone inference on debug frames
TORCH_COMPILE_DISABLE=1 python3 test_wan22_standalone.py \
    --model_path ./checkpoints/dreamzero_ur5e_wan22_lora \
    --output_dir ./checkpoints/ur5e_eval

# Watch the generated video
ffplay ./checkpoints/ur5e_eval/<timestamp>_standalone.mp4
```

A valid checkpoint directory contains: `config.json`, `adapter_model.safetensors`, `adapter_config.json`.

---

## Repo Changes for the UR5e Embodiment

| File | Change |
|---|---|
| `groot/vla/data/schema/embodiment_tags.py` | Added `EmbodimentTag.UR5E = "ur5e"` |
| `scripts/data/convert_lerobot_to_gear.py` | Added `"ur5e"` to `VALID_EMBODIMENT_TAGS` |
| `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml` | Added `modality_config_ur5e`, `transform_ur5e`, registered in all 4 global maps |
| `groot/vla/configs/data/dreamzero/ur5e_relative.yaml` | Dataset config for training |
| `scripts/train/ur5e_training.sh` | Training script (supports `BACKBONE=wan22\|wan21`) |
| `scripts/data/record_ur5e.py` | Data collection script (default IP: `100.80.196.7`) |

---

## Troubleshooting

**`No module named 'rtde_receive'`**
```bash
conda install -c conda-forge boost ncurses
pip install ur_rtde
```

**`pyzed not installed`**
```bash
source /home/d/miniconda3/bin/activate dreamzero
python /usr/local/zed/get_python_api.py
```

**`Cannot open camera device N`**
```bash
ls /dev/video*   # find correct index
# then set --left-camera-id / --wrist-camera-id accordingly
```

**`meta/embodiment.json missing` during training**
→ Run Step 2 (the GEAR conversion) first.

**CUDA out of memory during training**
→ Use `BACKBONE=wan22` (5B uses less VRAM) or reduce `per_device_train_batch_size`.

**RTDE connection refused**
→ Confirm RTDE is enabled on the pendant (Settings → System → Real-Time Data Exchange).  
→ Confirm the robot IP is `100.80.196.7` and your machine is on the same network.
