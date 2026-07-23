# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DreamZero is a research implementation of **World Action Models** — systems that jointly predict actions and future video frames, enabling zero-shot control of unseen robotic tasks. Based on the paper "World Action Models are Zero-shot Policies" (arXiv:2602.15922) from NVIDIA GEAR Lab.

Two checkpoints are supported:
- **DreamZero-DROID** (14B params, Wan2.1 backbone) — DROID dataset, broad zero-shot coverage
- **DreamZero-AgiBot** (5B params, Wan2.2 backbone) — efficient few-shot adaptation to new embodiments

## Environment Setup

```bash
conda create -n dreamzero python=3.11 && conda activate dreamzero
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu129
pip install flash-attn
# GB200 only: pip install transformer_engine tensorrt
```

The `[dev]` extra adds `pytest`, `black`, and `isort`. Use plain `-e .` to skip them.

Download checkpoints:
```bash
hf download GEAR-Dreams/DreamZero-DROID --repo-type model --local-dir ./checkpoints/DreamZero-DROID
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
hf download google/umt5-xxl --local-dir ./checkpoints/umt5-xxl
```

## Code Formatting

```bash
black .
isort .
```

No linting config (flake8/ruff/mypy) is present in the repo.

## Running Inference

**Wan2.1 server (14B, multi-GPU)** — also available as `dreamzero-server` after `pip install -e .`:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py --port 5000 --enable-dit-cache \
  --model-path ./checkpoints/DreamZero-DROID
```

**Wan2.2 server (5B, lower VRAM):**
```bash
bash launch_server_WAN22.sh
```

**Test the server** (network smoke test, not a unit test):
```bash
python test_client_AR.py --port 5000
```

**Real-robot deployment (UR5e):** `scripts/deploy/deploy_ur5e.py` executes served
action chunks on the lab UR5e (ur_rtde servoJ + URCap gripper). Safety-gated:
only (24,7) UR5e fine-tune chunks may move the robot; `--dry-run`/`--mock`/
`--test-motion`/`--gripper-test` modes for incremental verification. Both servers
accept an embodiment tag (`--embodiment_tag ur5e` / `--embodiment-tag ur5e`).
See `docs/UR5E_REAL_DEPLOYMENT.md` (deployment) and
`docs/UR5E_RTX5090_DATA_CAPTURE.md` (data recording with `scripts/data/record_ur5e.py`).

There are no pytest unit tests in this repo. `test_client_AR.py` is a manual integration client that requires a running server.

## Training

```bash
export NUM_GPUS=4
bash scripts/train/droid_training.sh          # Wan2.1 backbone
bash scripts/train/droid_training_wan22.sh    # Wan2.2 backbone
```

Training is Hydra-configured. Override params on the CLI:
```bash
python -m groot.vla.experiment.experiment \
  --config-name conf \
  training.learning_rate=1e-4 \
  data=droid_relative
```

## Data Processing

```bash
python download_droid_dataset.py   # with resume/retry support

# Convert DROID RLDS → LeRobot format
python scripts/data/convert_droid.py <raw_dir> <output_dir> \
  --keep-ranges-path <path> --filter-failed

# Convert LeRobot → GEAR metadata format (for new embodiments)
python scripts/data/convert_lerobot_to_gear.py \
  --dataset-path <path> --embodiment-tag <name>
```

## Architecture

### Model Stack

```
VLA (groot/vla/model/)
├── Backbone: Identity wrapper over Wan2.1-I2V-14B / Wan2.2-TI2V-5B
│   ├── WAN Video VAE        — encodes/decodes video frames
│   ├── WAN Video DiT        — diffusion transformer for video prediction
│   ├── Text Encoder (T5)    — language instruction embeddings
│   └── Image Encoder (CLIP) — visual observation embeddings
└── Action Head: WANPolicyHead (flow-matching diffusion)
    └── Jointly predicts: future video frames + action sequences
```

The key insight: the model is a video diffusion model finetuned to also predict robot actions, enabling zero-shot generalization through visual world modeling.

### Inference Pipeline

`socket_test_optimized_AR.py` runs a multi-GPU WebSocket server:
1. Receives image observation + language instruction from client
2. Runs DiT-cached video prediction across distributed GPUs
3. Returns predicted action chunk via WebSocket

The VRAM management module (`groot/vla/model/dreamzero/vram_management.py`) handles component offloading between CPU/GPU during inference to fit within GPU memory.

### Training Modes

- **Coupled (BASE)**: backbone and action head trained together
- **Decoupled**: action head trained separately (post-train fine-tuning for new embodiments)

Supported via `training.mode` in Hydra config.

### Adding a New Embodiment

1. Register an embodiment tag in `groot/vla/model/schema/embodiment_tags.py`
2. Create a data config YAML (see `groot/vla/configs/data/dreamzero/agibot_relative.yaml` as reference)
3. Map dataset columns to modalities (video, state, action, language)
4. Run `convert_lerobot_to_gear.py` to generate metadata
5. See `docs/DATASET_TO_GEAR_AND_TRAIN.md` for the full guide

## Configuration System

All training config is Hydra-based under `groot/vla/configs/`. Key config groups:
- `model/dreamzero/` — VLA model, backbone, action head
- `data/dreamzero/` — per-embodiment data configs
- `model/dreamzero/transform/` — data augmentation pipelines
- `deepspeed/` — ZeRO-2/3 configs for distributed training

Root config: `groot/vla/configs/conf.yaml`

## Key Files

| File | Role |
|------|------|
| `socket_test_optimized_AR.py` | Wan2.1 inference server (modify for server-side changes) |
| `socket_test_optimized_AR_WAN22.py` | Wan2.2 inference server |
| `groot/vla/experiment/experiment.py` | Training loop (VLATrainer, extends HF Trainer) |
| `groot/vla/experiment/base.py` | Base trainer utilities |
| `groot/vla/model/dreamzero/n1_5/sim_policy.py` | GrootSimPolicy — Tianshou policy interface for eval |
| `eval_utils/policy_server.py` | WebSocket policy server for simulation eval |
| `groot/vla/data/lerobot.py` | Primary dataset loader (LeRobot v2.0 format) |
| `scripts/deploy/deploy_ur5e.py` | UR5e real-robot deployment client (safety-gated execution) |
| `scripts/data/record_ur5e.py` | UR5e demonstration recorder + hardware classes (cameras, URCap gripper) |

## Wan2.2 vs Wan2.1 Differences

Wan2.2 is a 5B parameter model (vs 14B) with architectural changes documented in `docs/WAN22_BACKBONE.md`. The main differences affecting this codebase:
- Different attention mechanism
- Separate config/transform files with `_wan22` suffix
- Lower VRAM requirements allow single-GPU or fewer GPUs
- `launch_server_WAN22.sh` / `droid_training_wan22.sh` are the entry points
