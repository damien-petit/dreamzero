# Running 14B DreamZero-DROID Inference on a Single RTX 5090 (32 GB)

This guide documents the **low-VRAM inference mode** that makes the 14B DreamZero-DROID
model (Wan2.1 backbone) run on a single 32 GB consumer GPU. The default inference path
loads ~42 GB of bf16 weights (DiT ~28 GB + umt5-xxl ~11 GB + CLIP ~2.3 GB + VAE ~0.5 GB)
and assumes 2× H100/GB200 — it cannot fit on an RTX 5090. Note that multi-GPU mode does
not help here: it replicates the full model per GPU (CFG-parallel), it does not shard it.

Verified on: RTX 5090 (32.6 GB VRAM, Blackwell sm_120), 62 GB system RAM.

## Measured performance

| Metric | Value |
|---|---|
| VRAM idle (after load) | ~18.4 GB |
| VRAM peak (episode end, video save) | ~30.8 GB |
| Host RAM peak during checkpoint load | ~53 GB |
| Latency per 24-action chunk | ~2.6–3.0 s (~2.2 s diffusion, 8 DiT steps) |
| Text encoding (on language change only) | ~1.8 s |

## Requirements

- The standard setup from the main README (conda env, `pip install -e .`, flash-attn)
- Checkpoints downloaded to `./checkpoints/`:
  `DreamZero-DROID`, `Wan2.1-I2V-14B-480P`, `umt5-xxl`
- At least ~60 GB of system RAM free during model loading
- No TensorRT / Transformer Engine needed

## Launch the server

```bash
TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
  --standalone --nproc_per_node=1 socket_test_optimized_AR.py --port 5000 \
  --low-vram \
  --model-path ./checkpoints/DreamZero-DROID \
  --wan-path ./checkpoints/Wan2.1-I2V-14B-480P \
  --text-encoder-path ./checkpoints/umt5-xxl
```

`TORCH_COMPILE_DISABLE=1` is **mandatory on Blackwell GPUs** (sm_120 is above PyTorch's
supported compile range; without it, torch.compile crashes at startup and the
`fullgraph=True` scheduler compilations raise at runtime).

Expected startup log milestones (loading takes a few minutes):

```
skip_component_loading enabled (checkpoint provides DiT weights)
[FP8] Quantized 16.15B DiT block params to float8_e4m3fn
Enabling text encoder CPU offload
Text encoder CPU offload enabled; keeping text encoder on CPU.
INFO:websockets.server:server listening on 0.0.0.0:5000
```

## Test it (client)

In a second terminal:

```bash
conda activate dreamzero
python test_client_AR.py --port 5000
```

The client streams frames from the `debug_image/*.mp4` videos and resizes them to the
resolution the server advertises in its metadata (`image_resolution=(180, 320)`), so
the videos there can be any resolution. Note the current `debug_image/` clips are
640×480 UR5e captures (the original 320×180 DROID clips are in
`debug_image/original/`) — the DROID model will run on them but the actions are not
meaningful for the UR5e scene; restore the originals for a representative smoke test.

Expected: the first call takes ~5–10 s (streamed text encoding + warmup), subsequent
calls ~2.6–3.0 s with `Text Encoder 0.00 seconds` (prompt-embedding cache hit). Actions
come back as (24, 8) chunks; a rollout video is saved under
`checkpoints/real_world_eval_gen_<date>_<index>/` on episode reset.

Watch memory while it runs:

```bash
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

## What `--low-vram` does

`--low-vram` enables all of the following. Each is also individually controllable via
its env var (or the finer-grained flags `--fp8-dit` / `--offload-text-encoder`).

| Mechanism | Env var | Saving | Cost |
|---|---|---|---|
| FP8 DiT: the 40 transformer blocks' linears (16.15B params) stored + computed in `float8_e4m3fn` via `torch._scaled_mm`; embeddings/norms/modulation/head stay bf16 | `DZ_FP8_DIT=1` | ~16 GB | Quantization error (see caveats) |
| T5 CPU offload: umt5-xxl stays on CPU, weights stream to GPU per-module during encoding; prompt embeddings cached per instruction | `DZ_OFFLOAD_TEXT_ENCODER=1` | ~11 GB | ~1.8 s, only when the language instruction changes |
| CLIP CPU offload: image encoder visits the GPU only at episode resets | (part of `DZ_OFFLOAD_TEXT_ENCODER`) | ~2.3 GB | ~1 s per episode reset |
| CFG disabled: skips the unconditional branch, halving KV/cross-attn caches and diffusion compute | `DZ_CFG_SCALE=1.0` | ~6.5 GB + 2× faster | Quality: the paper evaluates with cfg=5 (see caveats) |
| bf16 instantiation: components are built directly in bf16 instead of fp32-then-cast (halves the host-RAM peak at load; inference only) | `DZ_BF16_INIT=1` | host RAM | none |
| Allocator defragmentation | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | ~2 GB | none |

Unconditional improvements that also help other machines:

- `VLA.from_pretrained` streams checkpoint shards one at a time instead of holding the
  full 46 GB state dict in RAM alongside the model.
- The redundant ~28 GB base Wan2.1 DiT load is skipped when the checkpoint already
  contains DiT weights (`skip_component_loading` auto-enabled).
- T5/CLIP/VAE `.pth` files are loaded with `mmap` (file-backed, evictable).

## Caveats

1. **Quality is unvalidated against the paper setup.** Two deviations: FP8 weights
   (vs bf16) and CFG off (vs cfg_scale=5). The rollout videos look coherent, but
   validate action quality on your robot. Both knobs are reversible for A/B testing:
   - `DZ_FP8_DIT=0` — full bf16 DiT (will NOT fit on one 5090; needs the VRAM back)
   - `DZ_CFG_SCALE=5.0` — re-enable guidance (will OOM near episode end on one 5090:
     the checkpoint uses global attention, so the pos+neg KV caches reach ~13 GB)
2. **Headroom is thin (~1.8 GB at peak).** Full episodes with video saving passed in
   testing. If OOM recurs, the next levers are shorter episodes (`num_frames`) or an
   sm_120 NVFP4 TensorRT engine rebuild (`scripts/inference/build_trt_engine.py` — the
   prebuilt engine in the checkpoint targets GB200 and will not load on the 5090).
3. **Do not save checkpoints from a low-VRAM server process** — the offload wrappers
   change the T5 state-dict key layout. Inference is unaffected.
4. Training is unaffected by all of this: every mechanism is gated behind env vars that
   only the inference server sets.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Process killed with `exitcode: -9` (SIGKILL) during load | Host RAM exhausted. Free RAM (need ~55 GB) or add swap; make sure you are on a build that includes the shard-streaming loader. |
| `torch.compile ... found no compiled frames` | `TORCH_COMPILE_DISABLE=1` missing from the environment, or an un-gated `fullgraph=True` compile was added. |
| `Expected all tensors to be on the same device` in the text encoder | Inputs were routed to CPU because device resolution used the offloaded T5; fixed in `WANPolicyHead.device` / `VLA.prepare_input` — make sure both changes are present. |
| CUDA OOM at episode end with `DZ_CFG_SCALE=5.0` | Expected on 32 GB — CFG needs ~6.5 GB more than the card has. Use `DZ_CFG_SCALE=1.0`. |
| First call very slow (~10 s) | Normal: streamed T5 encoding + CUDA warmup. Subsequent calls hit the prompt cache. |
| `VideoToTensor: ... has invalid resolution (640, 480), expected (320, 180)` | The client sent raw video frames without resizing. Fixed in `test_client_AR.py` (resizes to the server's advertised `image_resolution`) — update your client if you see this. |

## Alternative: Wan2.2 5B server (lower VRAM, no special flags)

The Wan2.2-TI2V-5B checkpoint (`checkpoints/dreamzero_droid_wan22_lora`) fits on the
5090 without any low-VRAM machinery and is the faster option:

```bash
bash launch_server_WAN22_v2.sh          # serves on port 5000, saves rollout videos
python test_client_AR.py --port 5000    # same client
```

`launch_server_WAN22_v2.sh` uses `eval_utils/serve_dreamzero_wan22.py`, which resets
the start frame between episodes and wraps VAE decode in `no_grad` (prefer it over the
older `launch_server_WAN22.sh`). To serve a different fine-tune (e.g. a UR5e LoRA),
change `--model_path` in the script.

Both servers accept an embodiment tag for non-DROID fine-tunes (`--embodiment_tag ur5e`
on the 5B server, `--embodiment-tag ur5e` on the 14B one). To execute the served policy
on the real UR5e — including test modes that need no fine-tune — see
[UR5E_REAL_DEPLOYMENT.md](UR5E_REAL_DEPLOYMENT.md).
