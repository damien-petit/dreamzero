import dataclasses
import logging
import socket
import asyncio
import os
import http
import time
import traceback
import torch
import tyro
from einops import rearrange
import datetime
import imageio
import numpy as np

import huggingface_hub
import transformers

# =========================================================================
# ADDED: MONKEY PATCH TO FORCE OFFLINE MODE FOR --local-dir CHECKPOINTS
# Intercepts hardcoded network calls and library API requests, forcefully
# redirecting them to your custom local checkpoint folders.
# =========================================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 1. Patch hf_hub_download for direct file fetches
_original_hf_hub_download = huggingface_hub.hf_hub_download

def offline_hf_hub_download(*args, **kwargs):
    repo_id = kwargs.get("repo_id") or (args[0] if len(args) > 0 else "")
    filename = kwargs.get("filename") or (args[1] if len(args) > 1 else "")
    
    if repo_id and filename:
        if "Wan" in repo_id:
            local_dir = os.path.abspath("./checkpoints/Wan2.1-I2V-14B-480P")
        elif "umt5" in repo_id:
            local_dir = os.path.abspath("./checkpoints/umt5-xxl")
        else:
            local_dir = os.path.abspath(f"./checkpoints/{repo_id.split('/')[-1]}")
            
        local_path = os.path.join(local_dir, filename)
        
        if os.path.exists(local_path):
            print(f"[OFFLINE PATCH] Intercepted network request. Serving local file: {local_path}")
            return local_path
            
    print(f"[OFFLINE PATCH] Warning: {filename} not found locally. Attempting network fallback...")
    return _original_hf_hub_download(*args, **kwargs)

huggingface_hub.hf_hub_download = offline_hf_hub_download

# 2. Patch transformers.AutoTokenizer for the evaluation transforms
_orig_auto_tok = transformers.AutoTokenizer.from_pretrained

@classmethod
def offline_auto_tok(cls, pretrained_model_name_or_path, *args, **kwargs):
    if "umt5" in str(pretrained_model_name_or_path):
        pretrained_model_name_or_path = os.path.abspath("./checkpoints/umt5-xxl")
        print(f"[OFFLINE PATCH] Redirected AutoTokenizer to local path: {pretrained_model_name_or_path}")
    return _orig_auto_tok(pretrained_model_name_or_path, *args, **kwargs)

transformers.AutoTokenizer.from_pretrained = offline_auto_tok
# =========================================================================

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
from tianshou.data import Batch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

# Use roboarena policy server interface
from eval_utils.policy_server import WebsocketPolicyServer as RoboarenaServer
from eval_utils.policy_server import PolicyServerConfig

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class Args:
    port: int = 5000
    timeout_seconds: int = 50000  # 10 hours default, configurable
    model_path: str = "./checkpoints/DreamZero-DROID"
    wan_path: str = "./checkpoints/Wan2.1-I2V-14B-480P"
    text_encoder_path: str = "./checkpoints/umt5-xxl"
    enable_dit_cache: bool = False
    index: int = 0
    max_chunk_size: int | None = None
    fp8_dit: bool = False
    offload_text_encoder: bool = False
    low_vram: bool = False  # implies fp8_dit + offload_text_encoder
    embodiment_tag: str = "oxe_droid"  # "oxe_droid" (DROID/Franka) or "ur5e"


# Modality key mappings: client observation keys -> model input keys per embodiment.
# Must match the served checkpoint's modality config (for ur5e:
# modality_config_ur5e in groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml).
VIDEO_KEY_MAPPING = {
    "oxe_droid": {
        "observation/exterior_image_0_left": "video.exterior_image_1_left",
        "observation/exterior_image_1_left": "video.exterior_image_2_left",
        "observation/wrist_image_left": "video.wrist_image_left",
    },
    # left USB cam -> slot 0, ZED 2i -> slot 1, wrist USB cam -> wrist
    "ur5e": {
        "observation/exterior_image_0_left": "video.left_camera",
        "observation/exterior_image_1_left": "video.right_camera",
        "observation/wrist_image_left": "video.wrist_camera",
    },
}
STATE_KEY_MAPPING = {
    "oxe_droid": ("state.joint_position", "state.gripper_position"),
    "ur5e": ("state.joint_position", "state.gripper_position"),
}
LANGUAGE_KEY_MAPPING = {
    "oxe_droid": "annotation.language.action_text",
    "ur5e": "annotation.task",
}
JOINT_DIM = {"oxe_droid": 7, "ur5e": 6}


class ARDroidRoboarenaPolicy:
    """Wrapper policy that implements roboarena.policy.BasePolicy interface for AR_droid."""
    
    FRAMES_PER_CHUNK = 4
    
    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        output_dir: str | None = None,
        embodiment_tag: str = "oxe_droid",
    ) -> None:
        self._policy = groot_policy
        self._signal_group = signal_group
        self._output_dir = output_dir
        self._embodiment_tag = (
            embodiment_tag if embodiment_tag in VIDEO_KEY_MAPPING else "oxe_droid"
        )
        self._video_key_mapping = VIDEO_KEY_MAPPING[self._embodiment_tag]
        self._state_keys = STATE_KEY_MAPPING[self._embodiment_tag]
        self._language_key = LANGUAGE_KEY_MAPPING[self._embodiment_tag]
        self._joint_dim = JOINT_DIM[self._embodiment_tag]

        self._frame_buffers: dict[str, list[np.ndarray]] = {
            model_key: [] for model_key in self._video_key_mapping.values()
        }
        self._call_count = 0
        self._is_first_call = True
        self._current_session_id: str | None = None
        self.video_across_time = []
        self._msg_index = 0
        
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
    
    def _convert_observation(self, obs: dict) -> dict:
        converted = {}
        for roboarena_key, droid_key in self._video_key_mapping.items():
            if roboarena_key in obs:
                data = obs[roboarena_key]
                if isinstance(data, np.ndarray):
                    if data.ndim == 4:
                        self._frame_buffers[droid_key].extend(list(data))
                    else:
                        self._frame_buffers[droid_key].append(data)

        num_frames = 1 if self._is_first_call else self.FRAMES_PER_CHUNK
        
        for droid_key, buffer in self._frame_buffers.items():
            if len(buffer) > 0:
                if len(buffer) >= num_frames:
                    frames_to_use = buffer[-num_frames:]
                else:
                    frames_to_use = buffer.copy()
                    while len(frames_to_use) < num_frames:
                        frames_to_use.insert(0, buffer[0])
                video = np.stack(frames_to_use, axis=0)
                converted[droid_key] = video
        
        state_joint_key, state_gripper_key = self._state_keys
        if "observation/joint_position" in obs:
            joint_pos = obs["observation/joint_position"]
            if joint_pos.ndim == 1:
                joint_pos = joint_pos.reshape(1, -1)
            converted[state_joint_key] = joint_pos.astype(np.float64)
        else:
            converted[state_joint_key] = np.zeros((1, self._joint_dim), dtype=np.float64)

        if "observation/gripper_position" in obs:
            gripper_pos = obs["observation/gripper_position"]
            if gripper_pos.ndim == 1:
                gripper_pos = gripper_pos.reshape(1, -1)
            converted[state_gripper_key] = gripper_pos.astype(np.float64)
        else:
            converted[state_gripper_key] = np.zeros((1, 1), dtype=np.float64)

        converted[self._language_key] = obs.get("prompt", "")
        return converted
    
    def _convert_action(self, action_dict: dict) -> np.ndarray:
        joint_action = None
        gripper_action = None
        
        for key, value in action_dict.items():
            if "joint_position" in key:
                joint_action = value
            elif "gripper_position" in key or "gripper" in key:
                gripper_action = value
        
        if joint_action is None:
            return np.zeros((1, 8), dtype=np.float32)
        
        if isinstance(joint_action, torch.Tensor):
            joint_action = joint_action.cpu().numpy()
        
        if joint_action.ndim == 1:
            joint_action = joint_action.reshape(1, -1)
        
        N = joint_action.shape[0]
        
        if gripper_action is not None:
            if isinstance(gripper_action, torch.Tensor):
                gripper_action = gripper_action.cpu().numpy()
            if gripper_action.ndim == 1:
                gripper_action = gripper_action.reshape(-1, 1)
            elif gripper_action.ndim == 0:
                gripper_action = gripper_action.reshape(1, 1)
        else:
            gripper_action = np.zeros((N, 1), dtype=np.float32)
        
        action = np.concatenate([joint_action, gripper_action], axis=-1).astype(np.float32)
        return action
    
    def _broadcast_batch_to_workers(self, obs: dict) -> None:
        import pickle
        serialized = pickle.dumps(obs)
        data_size = len(serialized)
        
        size_tensor = torch.tensor([data_size], dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)
        
        data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
        dist.broadcast(data_tensor, src=0)
    
    def infer(self, obs: dict) -> np.ndarray:
        session_id = obs.get("session_id", None)
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                logger.info(f"Session changed from '{self._current_session_id}' to '{session_id}', resetting state")
                self._reset_state()
            else:
                logger.info(f"New session started: '{session_id}'")
            self._current_session_id = session_id
        
        self._msg_index += 1
        self._call_count += 1
        
        converted_obs = self._convert_observation(obs)
        
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        dist.broadcast(signal_tensor, src=0, group=self._signal_group)
        
        self._broadcast_batch_to_workers(converted_obs)
        batch = Batch(obs=converted_obs)
        
        dist.barrier()
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
        dist.barrier()
        
        self.video_across_time.append(video_pred)
        
        action_chunk_dict = result_batch.act
        action_dict = {k: getattr(action_chunk_dict, k) for k in dir(action_chunk_dict) if k.startswith("action.")}
        action = self._convert_action(action_dict)
        
        if self._is_first_call:
            self._is_first_call = False
        
        return action
    
    def _reset_state(self, save_video: bool = True) -> None:
        if save_video and len(self.video_across_time) > 0 and self._output_dir:
            try:
                frame_list = []
                video_across_time_cat = torch.cat(self.video_across_time, dim=2)
                frames = self._policy.trained_model.action_head.vae.decode(
                    video_across_time_cat,
                    tiled=self._policy.trained_model.action_head.tiled,
                    tile_size=(self._policy.trained_model.action_head.tile_size_height, self._policy.trained_model.action_head.tile_size_width),
                    tile_stride=(self._policy.trained_model.action_head.tile_stride_height, self._policy.trained_model.action_head.tile_stride_width),
                )
                frames = rearrange(frames, "B C T H W -> B T H W C")[0]
                frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                frame_list.extend(frames)
                
                if len(frame_list) > 0:
                    sample_frame = frame_list[0]
                    if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                        save_dir = self._output_dir
                        os.makedirs(save_dir, exist_ok=True)
                        all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                        n = (len(frame_list) - 1) // 8
                        output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                        imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                        logger.info(f"Saved video on reset to: {output_path}")
            except Exception as e:
                logger.warning(f"Failed to save video on reset: {e}")
        
        for key in self._frame_buffers:
            self._frame_buffers[key] = []
        
        self._call_count = 0
        self._is_first_call = True
        self.video_across_time = []
    
    def reset(self, reset_info: dict) -> None:
        self._reset_state(save_video=True)


class WebsocketPolicyServer:
    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        output_dir: str | None = None,
        signal_group: dist.ProcessGroup | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._output_dir = output_dir
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        self.video_across_time = []
        self._msg_index = 0
        self._signal_group = signal_group
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            os.makedirs(os.path.join(self._output_dir, "inputs"), exist_ok=True)
    
    def serve_forever(self, rank: int = 0) -> None:
        asyncio.run(self.run(rank))

    async def run(self, rank: int = 0):
        if rank == 0:
            async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
                ping_interval=None,
            ) as server:
                await server.serve_forever()
        else:
            await self._worker_loop()

    async def _worker_loop(self):
        logger.info(f"Worker loop started for rank {dist.get_rank()}")
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        while True:
            try:
                dist.broadcast(signal_tensor, src=0, group=self._signal_group)
                signal = signal_tensor.item()
                if signal == 1:
                    logger.info(f"Rank {dist.get_rank()} received shutdown signal")
                    break
                elif signal == 2:
                    logger.info(f"Rank {dist.get_rank()} received idle signal. Waiting for next client.")
                    continue

                batch = self._receive_batch_from_rank0()
                dist.barrier()
                with torch.no_grad():
                    result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
                dist.barrier()

            except Exception as e:
                logger.error(f"Worker loop error on rank {dist.get_rank()}: {e}")
                traceback.print_exc()
                break

    def _receive_batch_from_rank0(self):
        import pickle
        size_tensor = torch.zeros(1, dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)
        data_size = size_tensor.item()

        data_tensor = torch.zeros(data_size, dtype=torch.uint8, device='cuda')
        dist.broadcast(data_tensor, src=0)

        obs = pickle.loads(data_tensor.cpu().numpy().tobytes())
        return Batch(obs=obs)

    def _broadcast_batch_to_workers(self, obs):
        import pickle
        serialized = pickle.dumps(obs)
        data_size = len(serialized)

        size_tensor = torch.tensor([data_size], dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)

        data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
        dist.broadcast(data_tensor, src=0)

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        
        try:
            while True:
                try:
                    start_time = time.perf_counter()
                    data = await websocket.recv()
                    recv_done = time.perf_counter()
                    obs = msgpack_numpy.unpackb(data)
                    print(f"Wait Time: {recv_done - start_time:.2f} seconds")
                    self._msg_index += 1

                    infer_start_time = time.perf_counter()

                    signal_tensor.zero_() 
                    dist.broadcast(signal_tensor, src=0, group=self._signal_group) 

                    self._broadcast_batch_to_workers(obs)
                    batch = Batch(obs=obs)

                    dist.barrier()
                    forward_start_time = time.perf_counter()
                    with torch.no_grad():
                        result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
                    dist.barrier()
                    print(f"Forward Time: {time.perf_counter() - forward_start_time:.2f} seconds")

                    action_chunk_dict = result_batch.act
                    video_chunk = video_pred

                    print(f"Inference Time: {time.perf_counter() - infer_start_time:.2f} seconds")
                    self.video_across_time.append(video_chunk)

                    self.video_across_time = []
                    
                    def batch_to_dict(batch):
                        out = {}
                        for k in dir(batch):
                            if not k.startswith("action."):
                                continue
                            out[k] = getattr(batch, k)
                        return out
                    
                    action_chunk_dict = batch_to_dict(action_chunk_dict)
                    await websocket.send(packer.pack(action_chunk_dict))

                except websockets.ConnectionClosed:
                    logger.info(f"Connection from {websocket.remote_address} closed")
                    self.video_across_time = []
                    break
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error.",
                    )
                    raise
        finally:
            logger.info(f"Rank 0: Client session ended. Sending idle signal (2) to workers.")
            signal_tensor.fill_(2)
            dist.broadcast(signal_tensor, src=0, group=self._signal_group)

def init_mesh() -> DeviceMesh:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) setting device to {rank}")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size, ),
        mesh_dim_names=("ip", ),
    )
    return mesh

def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main(args: Args) -> None:
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ["ATTENTION_BACKEND"] = "FA2"
    if args.low_vram:
        args.fp8_dit = True
        args.offload_text_encoder = True
        # Reduce allocator fragmentation; must be set before the first CUDA allocation.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # Disable CFG by default in low-VRAM mode: the pos+neg KV caches (~13 GB at
        # episode end with global attention) don't fit on 32GB alongside the weights.
        # Override with DZ_CFG_SCALE=5.0 if you have the headroom (e.g. 2 GPUs).
        os.environ.setdefault("DZ_CFG_SCALE", "1.0")
    os.environ["DZ_FP8_DIT"] = "1" if args.fp8_dit else "0"
    os.environ["DZ_OFFLOAD_TEXT_ENCODER"] = "1" if args.offload_text_encoder else "0"
    # Halve the host-RAM peak by instantiating components directly in bf16 (they are
    # cast to bf16 for inference anyway).
    os.environ["DZ_BF16_INIT"] = "1" if args.low_vram else os.environ.get("DZ_BF16_INIT", "0")
    torch._dynamo.config.recompile_limit = 800

    embodiment_tag = args.embodiment_tag
    if embodiment_tag not in VIDEO_KEY_MAPPING:
        raise SystemExit(
            f"Unknown embodiment_tag '{embodiment_tag}' — known: {list(VIDEO_KEY_MAPPING)}"
        )
    model_path = args.model_path
    policy_metadata = {
        "embodiment": embodiment_tag,
        "model_name": "dreamzero",
        "model_path": model_path,
    }

    device_mesh = init_mesh()
    rank = dist.get_rank()

    timeout_delta = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)
    logger.info(f"Rank {rank} initialized signal_group (gloo)")

    local_overrides = {
        "action_head_cfg.config.diffusion_model_cfg.diffusion_model_pretrained_path": args.wan_path,
        "action_head_cfg.config.vae_cfg.vae_pretrained_path": args.wan_path,
        "action_head_cfg.config.image_encoder_cfg.image_encoder_pretrained_path": args.wan_path,
        "action_head_cfg.config.text_encoder_cfg.text_encoder_pretrained_path": args.text_encoder_path,
    }

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
        model_config_overrides=local_overrides,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    if rank == 0:
        logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
        parent_dir = os.path.dirname(model_path)
        date_suffix = datetime.datetime.now().strftime("%Y%m%d")
        checkpoint_name = os.path.basename(model_path)
        output_dir = os.path.join(parent_dir, f"real_world_eval_gen_{date_suffix}_{args.index}", checkpoint_name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None
        logging.info(f"Rank {rank} starting as worker...")
    
    wrapper_policy = ARDroidRoboarenaPolicy(
        groot_policy=policy,
        signal_group=signal_group,
        output_dir=output_dir,
        embodiment_tag=embodiment_tag,
    )
    
    server_config = PolicyServerConfig(
        image_resolution=(180, 320), 
        needs_wrist_camera=True,
        n_external_cameras=2,
        needs_stereo_camera=False,
        needs_session_id=True,
        action_space="joint_position",
    )
    
    if rank == 0:
        roboarena_server = RoboarenaServer(
            policy=wrapper_policy,
            server_config=server_config,
            host="0.0.0.0",
            port=args.port,
        )
        roboarena_server.serve_forever()
    else:
        server = WebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata=policy_metadata,
            output_dir=output_dir,
            signal_group=signal_group,
        )
        asyncio.run(server._worker_loop())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    main(args)
