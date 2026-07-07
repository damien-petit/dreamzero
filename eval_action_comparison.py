#!/usr/bin/env python3
"""Compare action predictions across DreamZero checkpoints on the same debug_image inputs.

Runs each model sequentially on identical observations and reports:
  - Per-step MAE between every pair of models
  - Per-DOF mean absolute error
  - Cosine similarity of full action chunk vectors
  - Overall summary table

Usage:
    TORCH_COMPILE_DISABLE=1 python eval_action_comparison.py
"""
import os, sys, datetime, traceback, gc
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from tianshou.data import Batch

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29507")
if not dist.is_initialized():
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

torch._dynamo.config.recompile_limit = 800

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag

# ── checkpoints to compare ──────────────────────────────────────────────────
MODELS = {
    "wan21_14B_pretrained":  "./checkpoints/DreamZero-DROID",
    "wan22_r4_100steps":     "./checkpoints/dreamzero_droid_wan22_lora",
    "wan22_r32_5000steps":   "./checkpoints/dreamzero_droid_wan22_lora_r32/checkpoint-5000",
}

VIDEO_DIR = "./debug_image"
CAMERA_FILES = {
    "video.exterior_image_1_left": "exterior_image_1_left.mp4",
    "video.exterior_image_2_left": "exterior_image_2_left.mp4",
    "video.wrist_image_left":      "wrist_image_left.mp4",
}
PROMPT = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"
RELATIVE_OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON   = 24
NUM_CHUNKS       = 12   # keep short — comparison, not full episode
NATIVE_H, NATIVE_W = 180, 320   # raw debug_image resolution


# ── helpers ──────────────────────────────────────────────────────────────────
def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)

def get_model_resolution(model_path):
    """Read target H×W from config.json; fall back to native debug_image size."""
    import json
    cfg_path = os.path.join(model_path, "config.json")
    cfg = json.load(open(cfg_path))
    ah  = cfg.get("action_head_cfg", {}).get("config", {})
    h   = ah.get("target_video_height")
    w   = ah.get("target_video_width")
    if h and w:
        return int(h), int(w)
    return NATIVE_H, NATIVE_W   # 14B: no resize, pass native 180×320

def resize_frames(frames, target_h, target_w):
    if frames.shape[-3] == target_h and frames.shape[-2] == target_w:
        return frames
    if frames.ndim == 3:
        return cv2.resize(frames, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return np.stack([cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for f in frames])

def build_obs(camera_frames, frame_indices, num_frames, target_h=NATIVE_H, target_w=NATIVE_W):
    obs = {}
    for key, all_frames in camera_frames.items():
        sel = all_frames[frame_indices]
        obs[key] = resize_frames(sel[0] if num_frames == 1 else sel, target_h, target_w)
    obs["state.joint_position"]  = np.zeros((1, 7),  dtype=np.float64)
    obs["state.gripper_position"] = np.zeros((1, 1),  dtype=np.float64)
    obs["annotation.language.action_text"] = PROMPT
    return obs

def extract_actions(result_batch):
    """Return flat action chunk as numpy array (ACTION_HORIZON × action_dim)."""
    act = result_batch.act
    keys = [k for k in dir(act) if k.startswith("action.")]
    parts = []
    for k in sorted(keys):
        v = getattr(act, k)
        if torch.is_tensor(v):
            parts.append(v.float().cpu().numpy())
        elif isinstance(v, np.ndarray):
            parts.append(v.astype(np.float32))
    if not parts:
        return None
    # Normalise every part to 2D (horizon, dim):
    #   (batch, horizon, dim) → squeeze batch
    #   (horizon,)            → expand to (horizon, 1)   e.g. gripper_position
    #   (horizon, dim)        → keep
    chunks = []
    for p in parts:
        if p.ndim == 3:
            p = p.squeeze(0)
        if p.ndim == 1:
            p = p[:, np.newaxis]
        chunks.append(p)
    return np.concatenate(chunks, axis=-1)   # (horizon, total_dim)

def run_model(label, model_path, camera_frames, chunks_schedule):
    """Load model, run NUM_CHUNKS inference calls, return list of action arrays."""
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"  path: {model_path}")
    target_h, target_w = get_model_resolution(model_path)
    print(f"  resolution: {target_h}×{target_w}")
    device_mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))
    try:
        policy = GrootSimPolicy(
            embodiment_tag=EmbodimentTag("oxe_droid"),
            model_path=model_path,
            device="cuda",
            device_mesh=device_mesh,
        )
    except Exception as e:
        print(f"  FAILED to load: {e}")
        traceback.print_exc()
        return None

    actions = []
    # initial call
    obs   = build_obs(camera_frames, [0], num_frames=1, target_h=target_h, target_w=target_w)
    batch = Batch(obs=obs)
    try:
        with torch.no_grad():
            result, _ = policy.lazy_joint_forward_causal(batch)
        act = extract_actions(result)
        if act is not None:
            actions.append(act)
            print(f"  init call  | action shape={act.shape} | mean={act.mean():.4f} std={act.std():.4f}")
    except Exception as e:
        print(f"  init call FAILED: {e}")

    for i, frame_indices in enumerate(chunks_schedule):
        obs   = build_obs(camera_frames, frame_indices, num_frames=4, target_h=target_h, target_w=target_w)
        batch = Batch(obs=obs)
        try:
            with torch.no_grad():
                result, _ = policy.lazy_joint_forward_causal(batch)
            act = extract_actions(result)
            if act is not None:
                actions.append(act)
                if i % 4 == 0:
                    print(f"  chunk {i+1:2d}/{len(chunks_schedule)} | mean={act.mean():.4f} std={act.std():.4f}")
        except Exception as e:
            print(f"  chunk {i+1} FAILED: {e}")

    print(f"  collected {len(actions)} action chunks")
    del policy
    torch.cuda.empty_cache()
    gc.collect()
    return actions


def compare_pair(label_a, label_b, actions_a, actions_b):
    """Compute per-step and aggregate comparison metrics."""
    n = min(len(actions_a), len(actions_b))
    if n == 0:
        return
    maes, cosines = [], []
    for i in range(n):
        a = actions_a[i].flatten()
        b = actions_b[i].flatten()
        maes.append(np.abs(a - b).mean())
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        cosines.append(cos)

    all_a = np.concatenate([x.flatten() for x in actions_a[:n]])
    all_b = np.concatenate([x.flatten() for x in actions_b[:n]])
    per_dim_mae = np.abs(
        np.stack([x for x in actions_a[:n]]) - np.stack([x for x in actions_b[:n]])
    ).mean(axis=(0, 1))  # (action_dim,)

    print(f"\n── {label_a}  vs  {label_b} ──")
    print(f"  steps compared      : {n}")
    print(f"  mean MAE            : {np.mean(maes):.5f}  (std: {np.std(maes):.5f})")
    print(f"  mean cosine sim     : {np.mean(cosines):.5f}")
    print(f"  per-DOF MAE (first 10): {np.round(per_dim_mae[:10], 5).tolist()}")
    print(f"  max per-DOF MAE     : {per_dim_mae.max():.5f}  (DOF {per_dim_mae.argmax()})")
    return {"mae": np.mean(maes), "cosine": np.mean(cosines), "per_dof_mae": per_dim_mae}


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading debug frames...")
    camera_frames = {}
    for key, fname in CAMERA_FILES.items():
        camera_frames[key] = load_video(os.path.join(VIDEO_DIR, fname))
    total = min(v.shape[0] for v in camera_frames.values())
    print(f"  total frames: {total}")

    chunks_schedule = []
    cur = 23
    for _ in range(NUM_CHUNKS):
        idx = [max(cur + off, 0) for off in RELATIVE_OFFSETS]
        if idx[-1] >= total: break
        chunks_schedule.append(idx)
        cur += ACTION_HORIZON

    results = {}
    for label, path in MODELS.items():
        actions = run_model(label, path, camera_frames, chunks_schedule)
        results[label] = actions

    # ── pairwise comparison ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PAIRWISE ACTION COMPARISON")
    labels = [l for l, a in results.items() if a is not None and len(a) > 0]
    summary = {}
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            la, lb = labels[i], labels[j]
            m = compare_pair(la, lb, results[la], results[lb])
            if m:
                summary[(la, lb)] = m

    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print(f"  {'Pair':<50} {'MAE':>8} {'CosSim':>8}")
    print(f"  {'-'*50} {'-'*8} {'-'*8}")
    for (la, lb), m in summary.items():
        pair = f"{la[:22]} vs {lb[:22]}"
        print(f"  {pair:<50} {m['mae']:>8.5f} {m['cosine']:>8.5f}")

if __name__ == "__main__":
    main()
