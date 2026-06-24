#!/usr/bin/env python3
"""Test client for AR_droid policy server using roboarena interface.

Sends real video frames from debug_image/ directory instead of zero dummy images.

Expected server configuration:
    - image_resolution: (180, 320)
    - n_external_cameras: 2
    - needs_wrist_camera: True
    - action_space: "joint_position"

Usage:
    # Run this test to connect to the offline-configured server:
    python test_client_AR.py --host <server_host> --port 5000
"""

import argparse
import logging
import os
import time
import cv2
import numpy as np

import eval_utils.policy_server as policy_server
from eval_utils.policy_client import WebsocketClientPolicy

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "debug_image")

# roboarena key -> video filename
CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}

RELATIVE_OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON = 24


def load_all_frames(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from {video_path}")
    return np.stack(frames, axis=0)


def load_camera_frames() -> dict[str, np.ndarray]:
    camera_frames: dict[str, np.ndarray] = {}
    for cam_key, fname in CAMERA_FILES.items():
        path = os.path.join(VIDEO_DIR, fname)
        camera_frames[cam_key] = load_all_frames(path)
        logging.info(f"Loaded {cam_key}: {camera_frames[cam_key].shape}")
    return camera_frames


def build_frame_schedule(total_frames: int, num_chunks: int) -> list[list[int]]:
    chunks: list[list[int]] = []
    current_frame = 23  # first anchor frame
    for _ in range(num_chunks):
        indices = [max(current_frame + off, 0) for off in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            logging.info(
                f"Frame {indices[-1]} >= {total_frames}, stopping at {len(chunks)} chunks"
            )
            break
        chunks.append(indices)
        current_frame += ACTION_HORIZON
    return chunks


def _make_obs_from_video(
    camera_frames: dict[str, np.ndarray],
    frame_indices: list[int],
    prompt: str,
    session_id: str,
) -> dict:
    obs: dict = {}
    for cam_key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]
        if len(frame_indices) == 1:
            selected = selected[0]
        obs[cam_key] = selected

    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def _make_zero_observation(
    server_config: policy_server.PolicyServerConfig,
    prompt: str = "pick up the object",
    session_id: str | None = None,
) -> dict:
    obs = {}
    
    if server_config.image_resolution is not None:
        h, w = server_config.image_resolution
    else:
        h, w = 180, 320
    
    for i in range(server_config.n_external_cameras):
        obs[f"observation/exterior_image_{i}_left"] = np.zeros((h, w, 3), dtype=np.uint8)
        if server_config.needs_stereo_camera:
            obs[f"observation/exterior_image_{i}_right"] = np.zeros((h, w, 3), dtype=np.uint8)
    
    if server_config.needs_wrist_camera:
        obs["observation/wrist_image_left"] = np.zeros((h, w, 3), dtype=np.uint8)
        if server_config.needs_stereo_camera:
            obs["observation/wrist_image_right"] = np.zeros((h, w, 3), dtype=np.uint8)
    
    if server_config.needs_session_id:
        import uuid
        obs["session_id"] = session_id if session_id else str(uuid.uuid4())
    
    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    
    return obs


def test_ar_droid_policy_server(
    host: str = "localhost",
    port: int = 5000,
    num_chunks: int = 15,
    prompt: str = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan",
    use_zero_images: bool = False,
):
    logging.info(f"Connecting to AR_droid server at {host}:{port}...")
    
    client = WebsocketClientPolicy(host=host, port=port)
    metadata = client.get_server_metadata()
    logging.info(f"Server metadata: {metadata}")
    assert isinstance(metadata, dict), "Metadata should be a dict"
    
    try:
        server_config = policy_server.PolicyServerConfig(**metadata)
    except Exception as e:
        logging.error(f"Error parsing metadata: {e}")
        raise e
    
    logging.info(f"Server config: {server_config}")
    assert server_config.n_external_cameras == 2, f"Expected 2 external cameras, got {server_config.n_external_cameras}"
    assert server_config.needs_wrist_camera, "Expected wrist camera to be enabled"
    assert server_config.action_space == "joint_position", f"Expected joint_position action space, got {server_config.action_space}"
    
    import uuid
    session_id = str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}")

    if use_zero_images:
        logging.info("Using ZERO dummy images (legacy mode)")
        for i in range(num_chunks):
            obs = _make_zero_observation(server_config, prompt=prompt, session_id=session_id)
            logging.info(f"Inference {i + 1}/{num_chunks}: prompt='{prompt}'")
            t0 = time.time()
            actions = client.infer(obs)
            dt = time.time() - t0
            _log_action(actions, dt)

        logging.info("Sending reset...")
        client.reset({})
        logging.info("Done (zero-image mode).")
        return

    logging.info("Loading real video frames from debug_image/ directory")
    camera_frames = load_camera_frames()

    total_frames = min(v.shape[0] for v in camera_frames.values())
    logging.info(f"Total frames available: {total_frames}")

    chunks = build_frame_schedule(total_frames, num_chunks)

    logging.info("Frame schedule:")
    logging.info("  Initial: [0]")
    for i, indices in enumerate(chunks):
        logging.info(f"  Chunk {i}: {indices}")

    logging.info("=== Initial: frame [0] ===")
    obs = _make_obs_from_video(camera_frames, [0], prompt, session_id)
    t0 = time.time()
    actions = client.infer(obs)
    dt = time.time() - t0
    _log_action(actions, dt)

    for chunk_idx, frame_indices in enumerate(chunks):
        logging.info(f"=== Chunk {chunk_idx}: frames {frame_indices} ===")
        obs = _make_obs_from_video(camera_frames, frame_indices, prompt, session_id)
        t0 = time.time()
        actions = client.infer(obs)
        dt = time.time() - t0
        _log_action(actions, dt)

    logging.info("Sending reset to save video...")
    client.reset({})
    logging.info("Done.")


def _log_action(actions: np.ndarray, dt: float) -> None:
    assert isinstance(actions, np.ndarray), f"Expected numpy array, got {type(actions)}"
    assert actions.ndim == 2, f"Expected 2D array, got shape {actions.shape}"
    assert actions.shape[-1] == 8, (
        f"Expected 8 action dims (7 joints + 1 gripper), got {actions.shape[-1]}"
    )
    logging.info(
        f"  Action shape: {actions.shape}, "
        f"range: [{actions.min():.4f}, {actions.max():.4f}], "
        f"time: {dt:.2f}s"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test AR_droid policy server with real video frames from debug_image/"
    )
    parser.add_argument("--host", default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=15,
        help="Number of 4-frame chunks to send after the initial frame (default: 15)",
    )
    parser.add_argument(
        "--prompt",
        default="Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan",
        help="Language prompt for the policy",
    )
    parser.add_argument(
        "--use-zero-images",
        action="store_true",
        help="Use zero dummy images instead of real video frames (legacy mode)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    test_ar_droid_policy_server(
        host=args.host,
        port=args.port,
        num_chunks=args.num_chunks,
        prompt=args.prompt,
        use_zero_images=args.use_zero_images,
    )


if __name__ == "__main__":
    main()