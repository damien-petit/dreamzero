#!/usr/bin/env python3
"""Standalone WAN22 inference test — no server/client, generates video directly.

Loads DreamZero WAN22 model, runs causal inference on debug_image/ frames,
decodes and saves the predicted video.

Usage:
    conda activate dreamzero
    python test_wan22_standalone.py
"""

import os
import sys
import datetime
import traceback

os.environ["TORCH_COMPILE_DISABLE"] = "1"

import cv2
import imageio
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from tianshou.data import Batch
from einops import rearrange

# --- distributed setup (single GPU) ---
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29505")
if not dist.is_initialized():
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

torch._dynamo.config.recompile_limit = 800

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
import argparse
from groot.vla.data.schema import EmbodimentTag

_parser = argparse.ArgumentParser()
_parser.add_argument("--model_path", default="./checkpoints/dreamzero_droid_wan22_lora")
_parser.add_argument("--output_dir", default=None)
_args, _ = _parser.parse_known_args()

MODEL_PATH = _args.model_path
VIDEO_DIR = "./debug_image"
_default_out = "real_world_eval_gen_test/" + os.path.basename(MODEL_PATH.rstrip("/"))
OUTPUT_DIR = _args.output_dir or f"./checkpoints/{_default_out}"
PROMPT = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"

CAMERA_FILES = {
    "video.exterior_image_1_left": "exterior_image_1_left.mp4",
    "video.exterior_image_2_left": "exterior_image_2_left.mp4",
    "video.wrist_image_left": "wrist_image_left.mp4",
}

RELATIVE_OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON = 24
NUM_CHUNKS = 15
FRAMES_PER_CHUNK = 4


def load_video_frames(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def get_model_resolution(model_path):
    """Read target H×W from config.json; fall back to 14B native 180×320."""
    import json
    cfg_path = os.path.join(model_path, "config.json")
    try:
        cfg = json.load(open(cfg_path))
        ah = cfg.get("action_head_cfg", {}).get("config", {})
        h = ah.get("target_video_height")
        w = ah.get("target_video_width")
        if h and w:
            return int(h), int(w)
    except Exception:
        pass
    return 180, 320  # 14B native (no config resolution): pass native-ish size


TARGET_H, TARGET_W = get_model_resolution(MODEL_PATH)  # per-model resolution
print(f"Target resolution for {MODEL_PATH}: {TARGET_H}×{TARGET_W}")


def resize_frames(frames: np.ndarray) -> np.ndarray:
    """Resize (H,W,C) or (T,H,W,C) frames to TARGET_H x TARGET_W."""
    if frames.ndim == 3:
        if frames.shape[0] != TARGET_H or frames.shape[1] != TARGET_W:
            return cv2.resize(frames, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        return frames
    return np.stack(
        [cv2.resize(f, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR) for f in frames]
    )


def build_obs(camera_frames: dict, frame_indices: list, prompt: str, num_frames: int) -> dict:
    obs = {}
    for model_key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]
        if num_frames == 1:
            selected = resize_frames(selected[0])  # (H, W, C)
        else:
            selected = resize_frames(selected)  # (T, H, W, C)
        obs[model_key] = selected

    obs["state.joint_position"] = np.zeros((1, 7), dtype=np.float64)
    obs["state.gripper_position"] = np.zeros((1, 1), dtype=np.float64)
    obs["annotation.language.action_text"] = prompt
    return obs


def main():
    print("Loading WAN22 model...")
    device_mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("oxe_droid"),
        model_path=MODEL_PATH,
        device="cuda",
        device_mesh=device_mesh,
    )
    print("Model loaded.")

    print("Loading debug frames...")
    camera_frames = {}
    for model_key, fname in CAMERA_FILES.items():
        path = os.path.join(VIDEO_DIR, fname)
        frames = load_video_frames(path)
        camera_frames[model_key] = frames
        print(f"  {model_key}: {frames.shape}")

    total_frames = min(v.shape[0] for v in camera_frames.values())
    print(f"Total frames: {total_frames}")

    # Build frame schedule
    chunks = []
    current_frame = 23
    for _ in range(NUM_CHUNKS):
        indices = [max(current_frame + off, 0) for off in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            break
        chunks.append(indices)
        current_frame += ACTION_HORIZON

    print(f"Running {1 + len(chunks)} inference calls...")
    video_latents = []

    # Initial call: single frame
    obs = build_obs(camera_frames, [0], PROMPT, num_frames=1)
    batch = Batch(obs=obs)
    with torch.no_grad():
        result_batch, video_pred = policy.lazy_joint_forward_causal(batch)
    video_latents.append(video_pred.detach())
    print(f"  Initial: video_pred shape={video_pred.shape}, range=[{video_pred.min():.3f}, {video_pred.max():.3f}]")

    # Subsequent calls: 4 frames each
    for i, frame_indices in enumerate(chunks):
        obs = build_obs(camera_frames, frame_indices, PROMPT, num_frames=FRAMES_PER_CHUNK)
        batch = Batch(obs=obs)
        with torch.no_grad():
            result_batch, video_pred = policy.lazy_joint_forward_causal(batch)
        video_latents.append(video_pred.detach())
        if i % 5 == 0:
            print(f"  Chunk {i+1}/{len(chunks)}: video_pred shape={video_pred.shape}, range=[{video_pred.min():.3f}, {video_pred.max():.3f}]")

    print(f"Collected {len(video_latents)} latent tensors")

    # Decode video
    print("Decoding video latents through VAE...")
    action_head = policy.trained_model.action_head
    all_latents = torch.cat(video_latents, dim=2)
    print(f"  Combined latents shape: {all_latents.shape}")

    try:
        with torch.no_grad():
            frames = action_head.vae.decode(
                all_latents,
                tiled=action_head.tiled,
                tile_size=(action_head.tile_size_height, action_head.tile_size_width),
                tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
            )
        print(f"  Decoded frames shape: {frames.shape}")
        frames = rearrange(frames, "B C T H W -> B T H W C")[0]
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        print(f"  Pixel frames: shape={frames.shape}, mean={frames.mean():.1f}, std={frames.std():.1f}, min={frames.min()}, max={frames.max()}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
        output_path = os.path.join(OUTPUT_DIR, f"000000_{timestamp}_standalone.mp4")
        imageio.mimsave(output_path, list(frames), fps=5, codec="libx264")
        print(f"Saved video to: {output_path}")
        print(f"  {len(frames)} frames @ 160x320")

    except Exception as e:
        print(f"ERROR during decode: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
