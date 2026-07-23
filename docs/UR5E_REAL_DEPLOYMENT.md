# Deploying DreamZero on the Real UR5e (RTX 5090 machine)

`scripts/deploy/deploy_ur5e.py` closes the loop on the real robot: it streams
live camera frames and joint/gripper state to a running DreamZero inference
server and **executes** the returned action chunks on the UR5e (joints via
ur_rtde `servoJ`, Robotiq gripper via the URCap socket).

Related docs: [UR5E_RTX5090_DATA_CAPTURE.md](UR5E_RTX5090_DATA_CAPTURE.md)
(cameras, gripper wiring), [RTX5090_LOW_VRAM_INFERENCE.md](RTX5090_LOW_VRAM_INFERENCE.md)
(launching the servers).

---

## What model can move the robot

| Served checkpoint | Action shape | Robot execution |
|---|---|---|
| UR5e fine-tune (Wan2.2, `--embodiment_tag ur5e`) | (24, **7**) = 6 joints + gripper | **Yes** |
| DreamZero-DROID (14B or 5B) | (24, **8**) = 7 Franka joints + gripper | **Refused** — `--dry-run` only |

The DROID models predict 7-DoF Franka joints, which are structurally wrong for
the 6-joint UR5e; the client hard-refuses to execute them (see
[Testing joint commands without a fine-tune](#testing-joint-commands-without-a-fine-tune)
for the explicit override). Serve a UR5e fine-tune with either backbone:

```bash
# Wan2.2 5B backbone
TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python eval_utils/serve_dreamzero_wan22.py \
    --model_path ./checkpoints/dreamzero_ur5e_wan22_lora \
    --embodiment_tag ur5e \
    --port 5000 --save_video_pred

# Wan2.1 14B backbone (needs --low-vram on the 32 GB 5090)
TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
    --standalone --nproc_per_node=1 socket_test_optimized_AR.py --port 5000 \
    --low-vram --embodiment-tag ur5e \
    --model-path ./checkpoints/dreamzero_ur5e_wan21_lora \
    --wan-path ./checkpoints/Wan2.1-I2V-14B-480P \
    --text-encoder-path ./checkpoints/umt5-xxl
```

(Both servers carry `ur5e` entries in `VIDEO_KEY_MAPPING` / `STATE_KEY_MAPPING` /
`LANGUAGE_KEY_MAPPING` that map the client keys onto `video.left_camera` /
`video.right_camera` / `video.wrist_camera`, `state.*`, and `annotation.task`,
matching `modality_config_ur5e`. Wan2.1 chunks take ~2.6–3 s under `--low-vram`,
so pauses between motion bursts are longer than with the 5B.)

## Physical camera mapping (must match training data)

| Client obs key | Physical camera |
|---|---|
| `observation/exterior_image_0_left` | left USB cam (`DEFAULT_LEFT_CAMERA` by-path) |
| `observation/exterior_image_1_left` | ZED 2i |
| `observation/wrist_image_left` | wrist USB cam (`DEFAULT_WRIST_CAMERA` by-path) |

## Robot prerequisites

- Pendant: **Remote Control mode** (required by `RTDEControlInterface`;
  Settings → System → Remote Control, then select Remote on the top bar).
  RTDE (for reading state) must also be enabled.
- Arm powered on with brakes released — the URCap gripper socket answers `?`
  to every `GET` when the arm is off.
- **Speed slider ≤ 20 % for all first runs.** Operator within reach of the
  e-stop whenever the robot can move.

## How it runs (what to expect)

- Motion is **stop-and-go by design**: the client executes the first
  `--open-loop-horizon` (default 8) of each 24-action chunk at 15 Hz
  (~0.53 s of motion), then the robot holds while the server infers the next
  chunk (~1–3 s depending on model). This is inherent to inference latency.
- Observation history advances in *trajectory time*: history frames are
  recorded only while actions execute, so the model's 4-frame context looks
  like the continuous 15 Hz data it was trained on, and the newest frame is
  refreshed right before each inference.
- Returned joint values are **absolute targets in radians** (the server adds
  the current joint state you sent to the model's relative actions).
- Gripper actions ([0,1], 0=open) are binarized with hysteresis (close > 0.6,
  open < 0.4) and sent only on transitions; `--gripper-analog` retargets
  continuously instead.

## Command reference (`scripts/deploy/deploy_ur5e.py`)

One episode per invocation. `Ctrl-C` stops the robot cleanly at any time.

| Flag | Default | Meaning |
|---|---|---|
| **Connection / episode** | | |
| `--host` / `--port` | localhost / 5000 | Inference server address |
| `--robot-ip` | 100.80.196.7 | UR5e controller (RTDE + URCap gripper) |
| `--prompt` | required | Language instruction; fixed for the whole episode (changing it resets the model's context) |
| `--max-chunks` | 30 | Chunks per episode (each ≈ 0.53 s of motion at horizon 8) |
| `--open-loop-horizon` | 8 | Actions executed per 24-action chunk before re-inferring |
| `--joint-dim` | 6 | Joints sent in the observation: 6 = UR5e fine-tune, 7 = DROID (pads a zero) |
| **Cameras** | | |
| `--left-camera-id` / `--wrist-camera-id` | stable by-path devices | cv2 index or `/dev/v4l/by-path/...` path |
| `--zed-resolution` | HD720 | ZED capture resolution |
| `--resize-mode` | pad | `pad` = letterbox to 320×180 (matches sim eval); `stretch` = plain resize |
| **Modes** | | |
| `--mock` | off | Synthetic cameras/robot, implies `--dry-run` — protocol test without hardware |
| `--dry-run` | off | Full pipeline, prints chunk stats, **no** motion or gripper writes |
| `--step` | off | Operator confirmation (`go`) before **every** chunk, not just the first |
| `--gripper-test` | off | Standalone gripper activate + open/close cycle, then exit (no server, no arm motion) |
| `--test-motion` | off | Synthetic sine per joint through the full execution stack (no server) |
| `--test-motion-amplitude` | 0.15 rad | Sine amplitude; above ~0.19 also raise `--max-joint-delta` |
| `--allow-droid-execution` | off | Execute DROID (24,8) chunks' first 6 joint dims — pipeline test only; forces `--step`, 0.03 rad clamp, `--joint-dim 7` |
| **Safety** | | |
| `--max-joint-delta` | 0.05 rad | Per-step clamp (0.75 rad/s at 15 Hz) |
| `--abort-joint-delta` | 0.25 rad | Raw step above this aborts the episode |
| `--ramp-threshold` | 0.15 rad | Max distance to chunk start bridged by slow moveJ |
| `--joint-limits-json` | ±2π (elbow ±π) | Soft `[lo, hi]` per joint |
| `--lookahead` / `--gain` | 0.2 / 100 | servoJ smoothing (softest tracking) |
| **Gripper** | | |
| `--gripper-analog` | off | Continuous `SET POS` every step instead of binarized open/close with hysteresis |
| `--gripper-speed` / `--gripper-force` | 255 / 100 | Robotiq motion parameters set at activation |
| **Output** | | |
| `--out-dir` | runs/real | Episode artifacts root (`<out-dir>/<timestamp>/`) |
| `--no-save-video` | off | Skip writing `sent_frames.mp4` |
| `--no-reset-on-exit` | off | Skip the server reset (server then does not save its dream video) |

## Safety layer

| Mechanism | Default | Flag |
|---|---|---|
| Dry-run (no motion, no gripper writes) | off | `--dry-run` |
| Per-step joint delta clamp | 0.05 rad (≈0.75 rad/s) | `--max-joint-delta` |
| Abort if a raw step exceeds | 0.25 rad | `--abort-joint-delta` |
| Soft joint limits | ±2π (elbow ±π) | `--joint-limits-json` |
| Ramp-in to chunk start | slow moveJ if ≤ 0.15 rad, else abort | `--ramp-threshold` |
| Operator confirmation | before first motion (type `go`) | `--step` = every chunk |
| Watchdog | chunk stopped if a step overruns 250 ms | — |
| Guaranteed stop on exit | `servoStop`/`stopJ` in all paths (Ctrl-C, errors) | — |

## Testing joint commands without a fine-tune

Two modes exist to validate the execution stack before any UR5e checkpoint is
trained. Both require the pendant in Remote Control mode, slider ≤ 20 %, and an
operator at the e-stop.

**`--test-motion` (do this first)** — no policy server involved. Executes a
slow sine (default ±0.15 rad ≈ 8.6°, `--test-motion-amplitude`) on each joint
in turn (one joint per 24-step chunk, ending back at the start pose) through
the exact same code path as real deployment:
SafetyGuard clamping, `servoJ` timing, watchdog, logging, `sent_frames.mp4`.
Deterministic and model-free — this is the definitive "do joint commands work"
test.

```bash
python scripts/deploy/deploy_ur5e.py --test-motion --max-chunks 6   # one cycle per joint
```

**`--allow-droid-execution` (optional, model-in-the-loop)** — relaxes the
(N, 8) refusal and executes the first 6 of the DROID model's 7 Franka joint
deltas around the current pose. **The motion is semantically meaningless**: the
DROID model was trained only on 7-DoF Franka arms, and there is no valid mapping
onto UR5e joints — expect small arbitrary wiggles, nothing task-directed. Value:
end-to-end pipeline proof (camera → server → chunk → servoJ) before a fine-tune
exists. The flag forces `--step` (confirm every chunk), caps the per-step delta
at 0.03 rad, and forces `--joint-dim 7`.

```bash
python scripts/deploy/deploy_ur5e.py --prompt "test" \
    --allow-droid-execution --step --max-chunks 2
```

## Verification ladder (strictly in order)

```bash
# 1. Protocol only — no hardware (works against the DROID server too)
python scripts/deploy/deploy_ur5e.py --mock --joint-dim 7 --prompt "test" --max-chunks 2

# 2. Real sensors, no motion — robot on, remote control NOT needed
python scripts/deploy/deploy_ur5e.py --dry-run --joint-dim 7 --prompt "test" --max-chunks 2
#    then check runs/real/<timestamp>/sent_frames.mp4 (what the model saw)

# 3. Gripper only — activation + one open/close cycle, arm does not move
python scripts/deploy/deploy_ur5e.py --gripper-test

# 3.5 Joint-command test without a fine-tune (see section above)
python scripts/deploy/deploy_ur5e.py --test-motion --max-chunks 6
#    optionally: --allow-droid-execution against the DROID server

# 4. First motion — UR5e fine-tune served, speed slider <= 20%, hand on e-stop
python scripts/deploy/deploy_ur5e.py --prompt "your task" \
    --step --max-joint-delta 0.03 --max-chunks 2

# 5. Normal operation
python scripts/deploy/deploy_ur5e.py --prompt "your task"
```

Each episode writes `runs/real/<timestamp>/` with `sent_frames.mp4` (the exact
resized frames the model received) and `episode.jsonl` (per-step joint targets,
actual positions, gripper actions, inference latency). On exit the client calls
`reset`, which makes the server save its **dreamed video** — compare the two to
judge how well the world model tracked reality.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Server returned (N, 8) actions` refusal | You're serving a DROID checkpoint. Serve the UR5e fine-tune, or use `--dry-run`. |
| `RTDEControlInterface` fails to connect | Pendant not in Remote Control mode, or another control client is connected. |
| Gripper warning `invalid literal ... '?'` | Arm powered off / gripper unpowered — URCap answers `?` for every variable. Power the arm. |
| `ZED camera open failed: CAMERA STREAM FAILED TO START` | ZED's USB3 link didn't enumerate (only its HID interface visible in `lsusb`). Replug the ZED cable into its USB 3 port. |
| `Cannot open camera device` for a by-path device | Camera moved to a different physical USB port; see UR5E_RTX5090_DATA_CAPTURE.md. |
| Robot pauses much longer than ~3 s between bursts | Check server log; first call of a session includes text encoding (~2 s extra). |
| Chunk aborts with `raw joint step ... exceeds abort threshold` | Model commanded a jump — usually a bad fine-tune or wrong `--joint-dim`; investigate before loosening thresholds. |
