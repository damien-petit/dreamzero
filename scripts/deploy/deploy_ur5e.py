#!/usr/bin/env python3
"""
Real-robot deployment client: run a DreamZero policy on the UR5e.

Captures live camera frames and robot state, sends them to a running DreamZero
inference server (socket_test_optimized_AR.py or eval_utils/serve_dreamzero_wan22.py),
and EXECUTES the returned action chunks on the UR5e (joints via ur_rtde servoJ,
Robotiq gripper via the URCap socket server).

Safety model (READ docs/UR5E_REAL_DEPLOYMENT.md before first use):
  - Only (24, 7) action chunks (6 UR5e joints + gripper, i.e. a UR5e fine-tune)
    may move the robot. (24, 8) DROID/Franka chunks are dry-run only.
  - Per-step joint delta clamp + abort threshold + soft joint limits.
  - Operator must type 'go' before the first motion (--step: before every chunk).
  - servoStop/stopJ guaranteed on any exit path.

Verification ladder (in order):
  1. --mock --dry-run --joint-dim 7      # protocol only, no hardware
  2. --dry-run --joint-dim 7             # real sensors, no motion
  3. --gripper-test                      # gripper open/close cycle, no arm motion
  4. --step --max-joint-delta 0.03 --max-chunks 2   # first motion, speed slider <= 20%
  5. normal episodes

Example (UR5e fine-tune being served):
  python scripts/deploy/deploy_ur5e.py --port 5000 --prompt "pick up the red cube"
"""

import argparse
import json
import logging
import signal
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

from record_ur5e import (  # noqa: E402
    DEFAULT_LEFT_CAMERA,
    DEFAULT_WRIST_CAMERA,
    CV2Camera,
    MockCamera,
    MockRTDE,
    URCapGripper,
    ZedCamera,
    _camera_device,
)
from eval_utils.policy_client import WebsocketClientPolicy  # noqa: E402

try:
    from openpi_client import image_tools
    HAS_OPENPI_RESIZE = True
except ImportError:
    HAS_OPENPI_RESIZE = False

CONTROL_HZ = 15.0
DT = 1.0 / CONTROL_HZ
FRAME_OFFSETS = [-23, -16, -8, 0]  # obs history offsets, in executed control steps
OBS_KEYS = (
    "observation/exterior_image_0_left",   # left USB camera
    "observation/exterior_image_1_left",   # ZED 2i (right camera)
    "observation/wrist_image_left",        # wrist USB camera
)

log = logging.getLogger("deploy_ur5e")


class AbortMotion(Exception):
    """Raised when an action fails a safety check; stops the episode."""


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

class CameraDaemon:
    """Continuously drains a camera in a background thread so .latest() is
    always the current frame (V4L2 buffers otherwise serve stale frames after
    the multi-second inference pauses)."""

    def __init__(self, camera, name: str):
        self.camera = camera
        self.name = name
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                frame = self.camera.read()
            except Exception as e:
                log.error(f"{self.name}: camera read failed: {e}")
                time.sleep(0.1)
                continue
            with self._lock:
                self._latest = frame

    def latest(self) -> np.ndarray:
        with self._lock:
            if self._latest is None:
                raise RuntimeError(f"{self.name}: no frame captured yet")
            return self._latest

    def wait_for_frame(self, timeout: float = 5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest is not None:
                    return
            time.sleep(0.05)
        raise RuntimeError(f"{self.name}: no frame within {timeout}s")

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.camera.close()


class ObsHistory:
    """Ring buffer of observation frame-triples, one entry per EXECUTED control
    step (trajectory time, not wall clock). During the inference pause nothing
    is appended, so the [-23, -16, -8, 0] offsets keep the temporal meaning
    they had in training data recorded at 15 Hz."""

    def __init__(self, maxlen: int = 32):
        self._buf = deque(maxlen=maxlen)

    def append(self, triple: dict):
        self._buf.append(triple)

    def refresh_last(self, triple: dict):
        """Replace the newest entry so the offset-0 anchor frame is 'now'."""
        if self._buf:
            self._buf[-1] = triple
        else:
            self._buf.append(triple)

    def __len__(self):
        return len(self._buf)

    def build_obs_images(self, first_call: bool) -> dict:
        if first_call:
            return {k: self._buf[-1][k] for k in OBS_KEYS}
        stacks = {}
        n = len(self._buf)
        indices = [max(0, n - 1 + off) for off in FRAME_OFFSETS]
        for k in OBS_KEYS:
            stacks[k] = np.stack([self._buf[i][k] for i in indices], axis=0)
        return stacks


# ---------------------------------------------------------------------------
# Gripper
# ---------------------------------------------------------------------------

class GripperController:
    """Executes gripper actions. Default: binarize with hysteresis (close > 0.6,
    open < 0.4) and send SET POS only on state transitions — DROID-style policies
    emit near-binary gripper values and the 2F-85 stroke spans many control
    steps anyway. --gripper-analog retargets every step instead."""

    def __init__(self, gripper: URCapGripper | None, analog: bool = False,
                 dry_run: bool = False):
        self.gripper = gripper
        self.analog = analog
        self.dry_run = dry_run
        self._closed = False

    def read_norm(self) -> float:
        if self.gripper is None:
            return 0.0
        try:
            return float(self.gripper.get_current_position()) / 255.0
        except Exception as e:
            log.warning(f"gripper read failed ({e}); reporting 0.0")
            return 0.0

    def command(self, action: float):
        if self.gripper is None or self.dry_run:
            return
        if self.analog:
            self.gripper.set_position(int(round(np.clip(action, 0.0, 1.0) * 255)))
            return
        if action > 0.6 and not self._closed:
            self._closed = True
            self.gripper.set_position(255)
        elif action < 0.4 and self._closed:
            self._closed = False
            self.gripper.set_position(0)


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

class SafetyGuard:
    def __init__(self, max_joint_delta: float, abort_joint_delta: float,
                 limits_lo: np.ndarray, limits_hi: np.ndarray):
        self.max_joint_delta = max_joint_delta
        self.abort_joint_delta = abort_joint_delta
        self.limits_lo = limits_lo
        self.limits_hi = limits_hi

    def check_and_clamp(self, q_prev: np.ndarray, q_raw: np.ndarray) -> np.ndarray:
        delta = q_raw - q_prev
        worst = float(np.abs(delta).max())
        if worst > self.abort_joint_delta:
            raise AbortMotion(
                f"raw joint step {worst:.3f} rad exceeds abort threshold "
                f"{self.abort_joint_delta} rad — refusing to continue"
            )
        q_cmd = q_prev + np.clip(delta, -self.max_joint_delta, self.max_joint_delta)
        if np.any(q_cmd < self.limits_lo) or np.any(q_cmd > self.limits_hi):
            raise AbortMotion(
                f"target {np.round(q_cmd, 3).tolist()} outside soft joint limits"
            )
        return q_cmd


# ---------------------------------------------------------------------------
# Deployer
# ---------------------------------------------------------------------------

class UR5eDeployer:
    def __init__(self, args):
        self.args = args
        self.dt = DT
        self.horizon = args.open_loop_horizon
        self.session_id = str(uuid.uuid4())
        self.stop_requested = threading.Event()
        self.out_dir = Path(args.out_dir) / time.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_records = []
        self.chunk_idx = 0

        if args.test_motion:
            # Synthetic trajectories generated locally — no policy server involved.
            self.client = None
            self.img_h, self.img_w = 180, 320
            self.horizon = 24  # full sine period per chunk (returns to start pose)
            log.info("*** TEST-MOTION MODE — synthetic sine trajectories, no server ***")
        else:
            log.info(f"Connecting to policy server at {args.host}:{args.port} ...")
            self.client = WebsocketClientPolicy(host=args.host, port=args.port)
            meta = self.client.get_server_metadata()
            log.info(f"Server metadata: {meta}")
            self.img_h, self.img_w = meta.get("image_resolution") or (180, 320)

        if args.mock:
            log.info("*** MOCK MODE — no real hardware ***")
            self.rtde_r = MockRTDE()
            self.rtde_c = None
            cams = {
                OBS_KEYS[0]: MockCamera(480, 640, (60, 80, 60)),
                OBS_KEYS[1]: MockCamera(720, 1280, (60, 60, 100)),
                OBS_KEYS[2]: MockCamera(480, 640, (100, 60, 60)),
            }
            self.gripper = GripperController(None, args.gripper_analog, True)
        else:
            import rtde_receive
            log.info(f"Connecting to UR5e at {args.robot_ip} ...")
            self.rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
            self.rtde_c = None
            if not args.dry_run:
                import rtde_control
                self.rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
                log.info("RTDE control connected (robot WILL move).")
            try:
                g = URCapGripper(args.robot_ip)
                if not args.dry_run and not args.test_motion:
                    g.activate(speed=args.gripper_speed, force=args.gripper_force)
                log.info(f"Gripper connected, position {g.get_current_position()}.")
            except Exception as e:
                log.warning(f"Gripper unavailable ({e}); gripper channel will be 0.0")
                g = None
            # test-motion never writes to the gripper (dry_run=True for writes only)
            self.gripper = GripperController(
                g, args.gripper_analog, args.dry_run or args.test_motion
            )
            cams = {
                OBS_KEYS[0]: CV2Camera(args.left_camera_id, int(CONTROL_HZ)),
                OBS_KEYS[1]: ZedCamera(args.zed_resolution, int(CONTROL_HZ)),
                OBS_KEYS[2]: CV2Camera(args.wrist_camera_id, int(CONTROL_HZ)),
            }

        self.daemons = {k: CameraDaemon(cam, k) for k, cam in cams.items()}
        for d in self.daemons.values():
            d.wait_for_frame()
        self.history = ObsHistory()

        limits = np.array(json.loads(args.joint_limits_json), dtype=np.float64)
        self.guard = SafetyGuard(
            args.max_joint_delta, args.abort_joint_delta, limits[:, 0], limits[:, 1]
        )

    # -- observation building ------------------------------------------------

    def _resize(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.args.resize_mode == "pad" and HAS_OPENPI_RESIZE:
            return image_tools.resize_with_pad(rgb, self.img_h, self.img_w)
        return cv2.resize(rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)

    def grab_triple(self) -> dict:
        return {k: self._resize(d.latest()) for k, d in self.daemons.items()}

    def current_q(self) -> np.ndarray:
        return np.array(self.rtde_r.getActualQ(), dtype=np.float64)

    def build_obs(self, first_call: bool) -> dict:
        obs = self.history.build_obs_images(first_call)
        q = self.current_q()
        if self.args.joint_dim == 7:  # pad for DROID (Franka) dry-runs
            q = np.concatenate([q, [0.0]])
        obs["observation/joint_position"] = q.astype(np.float32)
        obs["observation/gripper_position"] = np.array(
            [self.gripper.read_norm()], dtype=np.float32
        )
        obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
        obs["prompt"] = self.args.prompt
        obs["session_id"] = self.session_id
        return obs

    def infer(self, first_call: bool) -> np.ndarray:
        obs = self.build_obs(first_call)
        t0 = time.monotonic()
        chunk = np.asarray(self.client.infer(obs))
        self.last_latency = time.monotonic() - t0
        log.info(f"infer: chunk {chunk.shape} in {self.last_latency:.2f}s")
        return chunk

    # -- action gating / stats ----------------------------------------------

    def gate_action_dims(self, chunk: np.ndarray):
        if chunk.ndim != 2 or chunk.shape[1] not in (7, 8):
            raise SystemExit(f"Unexpected action shape {chunk.shape}")
        if chunk.shape[1] == 8 and not self.args.dry_run:
            if self.args.allow_droid_execution:
                log.warning(
                    "EXECUTING DROID (Franka) ACTIONS ON THE UR5e: taking the first "
                    "6 of 7 joint dims. This motion is semantically MEANINGLESS "
                    "(different joint spaces) — pipeline test only."
                )
                return
            raise SystemExit(
                "Server returned (N, 8) actions: a 7-DoF Franka/DROID checkpoint.\n"
                "These cannot be executed on the 6-joint UR5e. Serve a UR5e "
                "fine-tune, use --dry-run to exercise the pipeline, or\n"
                "--allow-droid-execution for a clamped plumbing test."
            )

    def chunk_stats(self, chunk: np.ndarray) -> str:
        q = self.current_q()
        nj = min(chunk.shape[1] - 1, len(q))
        deltas = chunk[:, :nj] - q[:nj]
        return (
            f"chunk {self.chunk_idx}: {chunk.shape}, executing first {self.horizon}\n"
            f"  per-joint delta from current pose: min {np.round(deltas.min(0), 3)}\n"
            f"                                     max {np.round(deltas.max(0), 3)}\n"
            f"  gripper trace: {np.round(chunk[: self.horizon, -1], 2)}"
        )

    def confirm(self, prompt: str) -> bool:
        try:
            return input(f"{prompt} — type 'go' to proceed, anything else aborts: ") \
                .strip().lower() == "go"
        except EOFError:
            return False

    # -- execution ------------------------------------------------------------

    def ramp_in(self, target_q: np.ndarray):
        q = self.current_q()
        worst = float(np.abs(target_q - q).max())
        if worst <= self.args.max_joint_delta:
            return
        if worst <= self.args.ramp_threshold:
            log.info(f"ramp-in: moveJ {worst:.3f} rad to chunk start")
            self.rtde_c.moveJ(target_q.tolist(), 0.2, 0.5)
            return
        raise AbortMotion(
            f"chunk starts {worst:.3f} rad from current pose "
            f"(> ramp threshold {self.args.ramp_threshold}) — aborting"
        )

    def execute_chunk(self, chunk: np.ndarray):
        targets = chunk[: self.horizon]
        if self.args.dry_run:
            # No motion: simulate trajectory time so the history advances.
            for t in targets:
                self.history.append(self.grab_triple())
                time.sleep(self.dt)
                if self.stop_requested.is_set():
                    raise KeyboardInterrupt
            return

        self.ramp_in(targets[0][:6])
        q_prev = self.current_q()
        deadline = time.monotonic()
        try:
            for t in targets:
                if self.stop_requested.is_set():
                    raise KeyboardInterrupt
                if self.rtde_r.isProtectiveStopped():
                    raise AbortMotion("robot is in protective stop")
                q_cmd = self.guard.check_and_clamp(q_prev, t[:6])
                self.rtde_c.servoJ(q_cmd.tolist(), 0.0, 0.0, self.dt,
                                   self.args.lookahead, self.args.gain)
                self.gripper.command(float(t[-1]))
                self.history.append(self.grab_triple())
                self.log_records.append({
                    "chunk": self.chunk_idx,
                    "q_actual": self.current_q().tolist(),
                    "q_target_raw": t[:6].tolist(),
                    "q_target_cmd": q_cmd.tolist(),
                    "gripper_action": float(t[-1]),
                    "infer_latency": self.last_latency,
                })
                q_prev = q_cmd
                deadline += self.dt
                lag = time.monotonic() - deadline
                if lag > 0.25:
                    log.error(f"control step overran by {lag:.3f}s — stopping chunk")
                    break
                time.sleep(max(0.0, -lag))
        finally:
            self.rtde_c.servoStop(2.0)

    # -- synthetic test motion --------------------------------------------------

    def build_test_chunk(self, cycle: int) -> np.ndarray:
        """One full slow sine period (24 steps at 15 Hz) on a single joint,
        cycling through the joints; returns to the start pose each chunk."""
        joint = cycle % 6
        amp = self.args.test_motion_amplitude
        q0 = self.current_q()
        chunk = np.tile(np.concatenate([q0, [0.0]]), (24, 1))
        steps = np.sin(2.0 * np.pi * (np.arange(24) + 1) / 24.0) * amp
        chunk[:, joint] = q0[joint] + steps
        log.info(f"test-motion chunk {cycle}: joint {joint}, amplitude {amp} rad")
        return chunk

    # -- main loop -------------------------------------------------------------

    def run(self):
        if self.args.test_motion:
            self.last_latency = 0.0
            if not self.confirm(
                f"TEST MOTION: ±{self.args.test_motion_amplitude} rad sine on each "
                f"joint in turn, {self.args.max_chunks} chunks"
            ):
                log.info("Aborted by operator before any motion.")
                return
            while self.chunk_idx < self.args.max_chunks:
                self.execute_chunk(self.build_test_chunk(self.chunk_idx))
                self.chunk_idx += 1
            return

        self.history.append(self.grab_triple())
        chunk = self.infer(first_call=True)
        self.gate_action_dims(chunk)
        print(self.chunk_stats(chunk))

        if not self.args.dry_run:
            if not self.confirm("First motion"):
                log.info("Aborted by operator before any motion.")
                return

        while self.chunk_idx < self.args.max_chunks:
            if self.args.step and self.chunk_idx > 0 and not self.args.dry_run:
                print(self.chunk_stats(chunk))
                if not self.confirm(f"Chunk {self.chunk_idx}"):
                    break
            self.execute_chunk(chunk)
            self.chunk_idx += 1
            if self.chunk_idx >= self.args.max_chunks:
                break
            self.history.refresh_last(self.grab_triple())
            chunk = self.infer(first_call=False)
            self.gate_action_dims(chunk)

    # -- shutdown ---------------------------------------------------------------

    def shutdown(self):
        if self.rtde_c is not None:
            try:
                self.rtde_c.servoStop(2.0)
            except Exception:
                try:
                    self.rtde_c.stopJ(2.0)
                except Exception:
                    pass
            try:
                self.rtde_c.disconnect()
            except Exception:
                pass
        if not self.args.no_reset_on_exit and self.client is not None:
            try:
                self.client.reset({})  # server saves the dreamed video
                log.info("Server reset — dream video saved server-side.")
            except Exception as e:
                log.warning(f"server reset failed: {e}")
        self._save_artifacts()
        for d in self.daemons.values():
            try:
                d.stop()
            except Exception:
                pass
        if self.gripper.gripper is not None:
            self.gripper.gripper.close()

    def _save_artifacts(self):
        if not self.log_records and len(self.history) == 0:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "episode.jsonl", "w") as f:
            for rec in self.log_records:
                f.write(json.dumps(rec) + "\n")
        if not self.args.no_save_video and len(self.history) > 1:
            path = self.out_dir / "sent_frames.mp4"
            frames = list(self.history._buf)
            canvas0 = np.hstack([frames[0][k] for k in OBS_KEYS])
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), CONTROL_HZ,
                (canvas0.shape[1], canvas0.shape[0]),
            )
            for tr in frames:
                writer.write(cv2.cvtColor(np.hstack([tr[k] for k in OBS_KEYS]),
                                          cv2.COLOR_RGB2BGR))
            writer.release()
        log.info(f"Artifacts saved to {self.out_dir}")


# ---------------------------------------------------------------------------
# Gripper-only test
# ---------------------------------------------------------------------------

def gripper_test(args):
    g = URCapGripper(args.robot_ip)
    print(f"initial position: {g.get_current_position()}")
    g.activate(speed=args.gripper_speed, force=args.gripper_force)
    print("activated. closing ...")
    g.set_position(255)
    time.sleep(2.0)
    print(f"position after close: {g.get_current_position()}")
    print("opening ...")
    g.set_position(0)
    time.sleep(2.0)
    print(f"position after open: {g.get_current_position()}")
    g.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Deploy a DreamZero policy on the real UR5e",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--robot-ip", default="100.80.196.7")
    p.add_argument("--prompt", default=None, help="language instruction (required unless --gripper-test)")
    p.add_argument("--max-chunks", type=int, default=30)
    p.add_argument("--open-loop-horizon", type=int, default=8,
                   help="actions executed per chunk before re-inferring (of 24)")
    p.add_argument("--joint-dim", type=int, choices=(6, 7), default=6,
                   help="joints sent in observation: 6=UR5e fine-tune, 7=DROID dry-run (pads a zero)")
    p.add_argument("--left-camera-id", type=_camera_device, default=DEFAULT_LEFT_CAMERA)
    p.add_argument("--wrist-camera-id", type=_camera_device, default=DEFAULT_WRIST_CAMERA)
    p.add_argument("--zed-resolution", default="HD720", choices=["VGA", "HD720", "HD1080"])
    p.add_argument("--resize-mode", choices=("pad", "stretch"), default="pad")
    p.add_argument("--mock", action="store_true", help="synthetic cameras/robot, no hardware")
    p.add_argument("--dry-run", action="store_true", help="full pipeline, NO robot motion / gripper writes")
    p.add_argument("--step", action="store_true", help="require confirmation before every chunk")
    p.add_argument("--max-joint-delta", type=float, default=0.05,
                   help="per-step joint clamp [rad] (0.05 = 0.75 rad/s at 15 Hz)")
    p.add_argument("--abort-joint-delta", type=float, default=0.25,
                   help="abort episode if a raw step exceeds this [rad]")
    p.add_argument("--ramp-threshold", type=float, default=0.15,
                   help="max distance to chunk start bridged by slow moveJ [rad]")
    p.add_argument("--joint-limits-json",
                   default="[[-6.28,6.28],[-6.28,6.28],[-3.14,3.14],[-6.28,6.28],[-6.28,6.28],[-6.28,6.28]]",
                   help="soft [lo, hi] per joint, radians")
    p.add_argument("--lookahead", type=float, default=0.2, help="servoJ lookahead_time [0.03-0.2]")
    p.add_argument("--gain", type=int, default=100, help="servoJ gain [100-2000]")
    p.add_argument("--gripper-analog", action="store_true",
                   help="send continuous SET POS every step instead of binarized open/close")
    p.add_argument("--gripper-speed", type=int, default=255)
    p.add_argument("--gripper-force", type=int, default=100)
    p.add_argument("--gripper-test", action="store_true",
                   help="standalone gripper activate + open/close cycle, then exit")
    p.add_argument("--test-motion", action="store_true",
                   help="execute synthetic sine trajectories (one joint per chunk) through "
                        "the full execution stack — no policy server, validates joint commands")
    p.add_argument("--test-motion-amplitude", type=float, default=0.15,
                   help="sine amplitude for --test-motion [rad] (~8.6 deg; above ~0.19 "
                        "also raise --max-joint-delta or steps get clamped)")
    p.add_argument("--allow-droid-execution", action="store_true",
                   help="UNSAFE PLUMBING TEST: execute the first 6 joint dims of DROID "
                        "(N,8) Franka chunks; semantically meaningless motion; forces "
                        "--step and a 0.03 rad step clamp")
    p.add_argument("--out-dir", default="runs/real")
    p.add_argument("--no-save-video", action="store_true")
    p.add_argument("--no-reset-on-exit", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.gripper_test:
        gripper_test(args)
        return
    if not args.prompt and not args.test_motion:
        p.error("--prompt is required")
    if args.test_motion:
        args.prompt = args.prompt or "test motion"
    if args.mock:
        args.dry_run = True  # no control interface exists in mock mode
    if args.allow_droid_execution and not args.dry_run:
        if not args.step:
            p.error("--allow-droid-execution requires --step (confirm every chunk)")
        args.max_joint_delta = min(args.max_joint_delta, 0.03)
        if args.joint_dim != 7:
            log.warning("--allow-droid-execution: forcing --joint-dim 7 (DROID expects 7)")
            args.joint_dim = 7

    deployer = UR5eDeployer(args)
    signal.signal(signal.SIGTERM, lambda *_: deployer.stop_requested.set())
    try:
        deployer.run()
    except KeyboardInterrupt:
        log.info("Interrupted — stopping robot.")
    except AbortMotion as e:
        log.error(f"MOTION ABORTED: {e}")
    finally:
        deployer.shutdown()


if __name__ == "__main__":
    main()
