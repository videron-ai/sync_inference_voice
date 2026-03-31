#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Voice-controlled synchronous inference script using openwakeword + faster-whisper.

Fully local — no API keys required. Extends eval_sync.py with a background voice
listener process that:
  1. Continuously feeds 80 ms audio chunks to openwakeword for wake-word detection.
  2. On detection, records the following command utterance (energy-based VAD).
  3. Transcribes the command locally with faster-whisper.
  4. Fuzzy-matches the transcript to TASK_MAP, then updates current_task live.

Keyboard task switching (from eval_sync.py) remains available alongside voice.

Dependencies (install in your Docker environment):
    pip install openwakeword faster-whisper pyaudio numpy

Wake word models:
    Pre-trained names: "hey_jarvis", "alexa", "hey_mycroft", "hey_rhasspy"
    Custom models: pass a path to an ONNX or TFLite file via --wakeword_model

Usage:
    python eval_sync_oww.py \\
        --policy.path=<hf-repo-or-local-dir> \\
        --policy.device=cuda \\
        --robot.type=so101_follower \\
        --robot.port=/dev/follower \\
        --robot.id=videron_follower \\
        --robot.cameras="{desk_cam: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, gripper: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}}" \\
        --task="Put all the legos on the table in the blue bowl" \\
        --robot.calibration_dir=/Experiments \\
        --duration=1000 \\
        --visualize=true \\
        --compress_images=true \\
        --wakeword_model=hey_jarvis \\
        --wakeword_threshold=0.5 \\
        --stt_model=small \\
        --stt_device=cpu \\
        --stt_compute_type=int8
"""

import logging
import multiprocessing
import queue
import select
import sys
import time
from dataclasses import dataclass, field
from difflib import get_close_matches

import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    so_follower,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- TASK CONFIGURATION ---
TASK_MAP = {
    "1": "Put the red lego in the blue bowl",
    "2": "Put all the legos on the table in the blue bowl",
    "3": "Move the robot to the home position",
}
current_task = TASK_MAP["1"]


# ---------------------------------------------------------------------------
# Keyboard task switching (preserved from eval_sync.py)
# ---------------------------------------------------------------------------

def check_for_input(current_val: str) -> str:
    """Non-blocking stdin check for task switching."""
    if select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline().strip()
        if line in TASK_MAP:
            new_task = TASK_MAP[line]
            print(f">>> KEYBOARD SWITCH: {new_task}")
            return new_task
        else:
            print(f"Unknown command '{line}'. Available: {list(TASK_MAP.keys())}")
    return current_val


# ---------------------------------------------------------------------------
# Task matching from transcription
# ---------------------------------------------------------------------------

def match_task(transcription: str) -> str:
    """
    Map a transcription string to a task.

    Priority:
      1. Exact numeric key ("1", "2", "3").
      2. Fuzzy match against TASK_MAP values (difflib, cutoff=0.35).
      3. Raw transcription as a free-form task instruction.
    """
    text = transcription.strip()

    if text in TASK_MAP:
        return TASK_MAP[text]

    candidates = list(TASK_MAP.values())
    matches = get_close_matches(text, candidates, n=1, cutoff=0.35)
    if matches:
        return matches[0]

    # No close match — use the raw transcription as a task instruction.
    return text


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16000      # Hz — required by openwakeword and Whisper
_OWW_CHUNK = 1280         # samples (80 ms) — openwakeword's preferred chunk size
_SPEECH_THRESHOLD = 300   # RMS above this → speech onset during command recording
_SILENCE_THRESHOLD = 200  # RMS below this → silence during command recording
_SILENCE_CHUNKS = 20      # ~1.3 s of trailing silence ends an utterance
_RECORD_MAX_SECONDS = 6  # hard cap per command utterance


def _find_device_index(pa, name_substring: str) -> int | None:
    """Return the first PyAudio input device index whose name contains `name_substring`."""
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0 and name_substring.lower() in info["name"].lower():
            return i
    return None


def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))


def _record_command(stream, shutdown_event: multiprocessing.Event) -> np.ndarray | None:
    """
    Record a command utterance after wake-word detection.

    Waits up to _RECORD_MAX_SECONDS for speech to begin, then records until
    _SILENCE_CHUNKS consecutive silent chunks are observed. Returns a
    concatenated int16 numpy array, or None if nothing was captured.
    """
    frames: list[np.ndarray] = []
    silence_count = 0
    speech_started = False
    max_chunks = int(_RECORD_MAX_SECONDS * _SAMPLE_RATE / _OWW_CHUNK)

    for _ in range(max_chunks):
        if shutdown_event.is_set():
            return None
        raw = stream.read(_OWW_CHUNK, exception_on_overflow=False)
        chunk = np.frombuffer(raw, dtype=np.int16)
        rms = _rms(chunk)

        if not speech_started:
            if rms >= _SPEECH_THRESHOLD:
                speech_started = True
                frames.append(chunk)
            # Still waiting for speech to begin — skip silent preamble
        else:
            frames.append(chunk)
            if rms < _SILENCE_THRESHOLD:
                silence_count += 1
            else:
                silence_count = 0
            if silence_count >= _SILENCE_CHUNKS:
                break

    return np.concatenate(frames) if frames else None


# ---------------------------------------------------------------------------
# Voice listener process
# ---------------------------------------------------------------------------

def _voice_process_main(
    wakeword_model: str,
    wakeword_threshold: float,
    stt_model_size: str,
    stt_device: str,
    stt_compute_type: str,
    mic_device_name: str | None,
    task_queue: multiprocessing.Queue,
    shutdown_event: multiprocessing.Event,
    task_map: dict,
) -> None:
    """
    Runs in a child process, fully isolated from the main inference loop.

    Continuously reads audio from the microphone and:
      - Feeds 80 ms chunks to openwakeword for wake-word detection.
      - On wake-word trigger, resets the OWW model and records the command.
      - Transcribes the command with faster-whisper (local, no API key).
      - Matches the transcription to TASK_MAP and puts the result on task_queue.
    """
    logging.basicConfig(level=logging.INFO)
    proc_logger = logging.getLogger("voice_process")

    # --- Import audio libraries (managed externally) ---
    try:
        import pyaudio
    except ImportError:
        proc_logger.error("[VOICE] pyaudio not found. Install: pip install pyaudio")
        return

    try:
        from openwakeword.model import Model as OWWModel
    except ImportError:
        proc_logger.error("[VOICE] openwakeword not found. Install: pip install openwakeword")
        return

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        proc_logger.error("[VOICE] faster-whisper not found. Install: pip install faster-whisper")
        return

    # --- Load openwakeword ---
    proc_logger.info(f"[VOICE] Loading openwakeword model: '{wakeword_model}'")
    try:
        #import openwakeword
        #import os
        jarvis_model_path = "/usr/local/lib/python3.12/dist-packages/openwakeword/resources/models/hey_jarvis_v0.1.onnx"

        #from openwakeword.model import Model as OWWModel
        oww_model = OWWModel(wakeword_model_paths=[jarvis_model_path])
        
    except Exception as exc:
        proc_logger.error(f"[VOICE] Failed to load openwakeword model: {exc}")
        return

    wakeword_names = list(oww_model.models.keys())
    proc_logger.info(f"[VOICE] openwakeword ready — classes: {wakeword_names}, threshold: {wakeword_threshold}")

    # --- Load faster-whisper ---
    proc_logger.info(f"[VOICE] Loading faster-whisper model: '{stt_model_size}' on {stt_device} ({stt_compute_type})")
    try:
        stt_model = WhisperModel(
            stt_model_size,
            device=stt_device,
            compute_type=stt_compute_type,
        )
    except Exception as exc:
        proc_logger.error(f"[VOICE] Failed to load faster-whisper model: {exc}")
        return

    proc_logger.info("[VOICE] faster-whisper ready")

    # --- Open microphone stream ---
    pa = pyaudio.PyAudio()
    if mic_device_name:
        device_index = _find_device_index(pa, mic_device_name)
        if device_index is None:
            proc_logger.warning(
                f"[VOICE] Could not find input device matching '{mic_device_name}'. "
                "Falling back to system default microphone."
            )
        else:
            proc_logger.info(f"[VOICE] Using input device index {device_index} ('{mic_device_name}')")
    else:
        device_index = None
        proc_logger.info("[VOICE] No mic device name specified — using system default microphone.")
    try:
        stream = pa.open(
            rate=_SAMPLE_RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=_OWW_CHUNK,
            input_device_index=device_index,
        )
    except Exception as exc:
        proc_logger.error(f"[VOICE] Failed to open microphone: {exc}")
        pa.terminate()
        return

    proc_logger.info("[VOICE] Microphone open — listening for wake word …")

    try:
        while not shutdown_event.is_set():
            # --- Read one 80 ms chunk ---
            raw = stream.read(_OWW_CHUNK, exception_on_overflow=False)
            chunk = np.frombuffer(raw, dtype=np.int16)

            # --- Wake-word detection ---
            # predict() returns {class_name: score, ...} for all loaded models
            predictions: dict[str, float] = oww_model.predict(chunk)

            triggered = any(score >= wakeword_threshold for score in predictions.values())
            if not triggered:
                continue

            best = max(predictions, key=predictions.get)
            proc_logger.info(
                f"[VOICE] Wake word detected: '{best}' (score={predictions[best]:.3f})"
            )
            task_menu = "\n".join(f"  [{k}] {v}" for k, v in task_map.items())
            print(
                f"\n>>> WAKE WORD DETECTED — speak your task now <<<\n"
                f"Available tasks:\n{task_menu}\n",
                flush=True,
            )

            # Reset OWW state so the same phrase doesn't immediately re-trigger
            oww_model.reset()

            # --- Record the command utterance ---
            command_audio = _record_command(stream, shutdown_event)
            if command_audio is None or len(command_audio) == 0:
                proc_logger.info("[VOICE] No command audio captured — resuming listening")
                continue

            # --- Transcribe with faster-whisper ---
            # Convert int16 PCM → float32 normalised to [-1, 1]
            audio_f32 = command_audio.astype(np.float32) / 32768.0

            try:
                segments, info = stt_model.transcribe(
                    audio_f32,
                    language="en",
                    beam_size=5,
                    vad_filter=True,
                )
                transcription = " ".join(seg.text for seg in segments).strip()
            except Exception as exc:
                proc_logger.warning(f"[VOICE] Transcription error: {exc}")
                continue

            if not transcription:
                proc_logger.info("[VOICE] Empty transcription — resuming listening")
                continue

            proc_logger.info(
                f"[VOICE] Transcribed: '{transcription}' "
                f"(lang={info.language}, conf={info.language_probability:.2f})"
            )

            # --- Match to task and enqueue ---
            new_task = match_task(transcription)
            task_queue.put(new_task)
            print(f">>> VOICE SWITCH: {new_task}", flush=True)

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        proc_logger.info("[VOICE] Process shut down.")


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SyncOWWConfig(HubMixin):
    """Configuration for openwakeword + faster-whisper voice-controlled inference."""

    policy: PreTrainedConfig | None = None
    robot: RobotConfig | None = None
    duration: float = 30.0
    fps: float = 30.0
    task: str = field(default="", metadata={"help": "Default task to execute at startup"})

    # Rerun visualization
    visualize: bool = False
    rerun_session_name: str = "eval_sync_oww"
    rerun_ip: str | None = None
    rerun_port: int | None = None
    compress_images: bool = False

    # Wake-word detection (openwakeword)
    wakeword_model: str = field(
        default="hey_jarvis",
        metadata={
            "help": (
                "openwakeword model name or path to a custom ONNX/TFLite file. "
                "Pre-trained names: hey_jarvis, alexa, hey_mycroft, hey_rhasspy. "
                "Download pre-trained models with: python -c \"import openwakeword; openwakeword.utils.download_models()\""
            )
        },
    )
    wakeword_threshold: float = field(
        default=0.5,
        metadata={"help": "Confidence threshold (0–1) for wake-word detection. Lower = more sensitive."},
    )

    # Speech-to-text (faster-whisper)
    stt_model: str = field(
        default="small",
        metadata={"help": "faster-whisper model size: tiny, base, small, medium, large-v3, turbo"},
    )
    stt_device: str = field(
        default="cpu",
        metadata={"help": "Device for faster-whisper: cpu or cuda. Defaults to cpu to avoid GPU contention."},
    )
    stt_compute_type: str = field(
        default="int8",
        metadata={"help": "Compute type for faster-whisper: int8 (fast/light), float16, float32"},
    )

    # Microphone
    mic_device_name: str | None = field(
        default="Logi USB Headset: Audio",
        metadata={"help": "Substring to match against PyAudio input device names. None uses the system default."},
    )

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            raise ValueError("Policy path is required (--policy.path=...)")

        if self.robot is None:
            raise ValueError("Robot configuration must be provided")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@parser.wrap()
def main(cfg: SyncOWWConfig):
    init_logging()
    logger.info("Starting openwakeword + faster-whisper voice-controlled inference")

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    # --- Load policy ---
    logger.info(f"Loading policy from {cfg.policy.pretrained_path}")
    policy_class = get_policy_class(cfg.policy.type)
    config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)

    if config.use_peft:
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(cfg.policy.pretrained_path)
        policy = policy_class.from_pretrained(
            pretrained_name_or_path=peft_config.base_model_name_or_path, config=config
        )
        policy = PeftModel.from_pretrained(policy, cfg.policy.pretrained_path, config=peft_config)
    else:
        policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=config)

    policy = policy.to(cfg.policy.device)
    policy.eval()
    policy.reset()
    logger.info(f"Policy loaded on {cfg.policy.device}")

    # --- Load pre/post processors ---
    logger.info(f"Loading processors from {cfg.policy.pretrained_path}")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=None,
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
        },
    )
    logger.info("Processors loaded")

    # --- Connect robot ---
    logger.info(f"Connecting robot: {cfg.robot.type}")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    robot_observation_processor = make_default_robot_observation_processor()
    robot_action_processor = make_default_robot_action_processor()
    dataset_features = hw_to_dataset_features(robot.observation_features, "observation")

    if cfg.visualize:
        init_rerun(
            session_name=cfg.rerun_session_name,
            ip=cfg.rerun_ip,
            port=cfg.rerun_port,
        )
        logger.info("Rerun visualization initialized")

    global current_task
    if cfg.task:
        current_task = cfg.task

    # --- Start voice listener in a child process ---
    # A separate process keeps audio I/O and model inference isolated from the
    # main loop's real-time constraints.
    task_queue: multiprocessing.Queue = multiprocessing.Queue()
    voice_shutdown: multiprocessing.Event = multiprocessing.Event()

    voice_proc = multiprocessing.Process(
        target=_voice_process_main,
        args=(
            cfg.wakeword_model,
            cfg.wakeword_threshold,
            cfg.stt_model,
            cfg.stt_device,
            cfg.stt_compute_type,
            cfg.mic_device_name,
            task_queue,
            voice_shutdown,
            TASK_MAP,
        ),
        daemon=True,
        name="voice-listener",
    )
    voice_proc.start()
    logger.info(
        f"Voice process started (PID {voice_proc.pid}) — "
        f"wake word model: '{cfg.wakeword_model}', threshold: {cfg.wakeword_threshold}, "
        f"STT: faster-whisper '{cfg.stt_model}' on {cfg.stt_device}"
    )

    action_interval = 1.0 / cfg.fps
    start_time = time.time()
    step = 0

    logger.info(f"Running for {cfg.duration}s at {cfg.fps} Hz — task: '{current_task}'")
    logger.info(f"Keyboard: type {list(TASK_MAP.keys())} + Enter to switch tasks")
    logger.info(f"Voice: say '{cfg.wakeword_model}' wake word, then speak your task")

    try:
        while not shutdown_event.is_set() and (time.time() - start_time) < cfg.duration:
            t0 = time.perf_counter()

            # Non-blocking keyboard task switch
            current_task = check_for_input(current_task)

            # Drain voice task queue — keep only the most recent command
            try:
                while True:
                    current_task = task_queue.get_nowait()
            except queue.Empty:
                pass

            active_task = current_task

            # --- Observation ---
            obs = robot.get_observation()

            if cfg.visualize:
                import rerun as rr
                rr.set_time_sequence("step", step)
                log_rerun_data(observation=obs, compress_images=cfg.compress_images)

            obs_processed = robot_observation_processor(obs)
            obs_dict = build_dataset_frame(dataset_features, obs_processed, prefix="observation")

            for name in obs_dict:
                obs_dict[name] = torch.from_numpy(obs_dict[name])
                if "image" in name:
                    obs_dict[name] = obs_dict[name].type(torch.float32) / 255
                    obs_dict[name] = obs_dict[name].permute(2, 0, 1).contiguous()
                obs_dict[name] = obs_dict[name].unsqueeze(0).to(cfg.policy.device)

            obs_dict["task"] = [active_task]
            obs_dict["robot_type"] = robot.name if hasattr(robot, "name") else ""

            # --- Preprocess ---
            batch = preprocessor(obs_dict)

            # --- Synchronous inference ---
            action = policy.select_action(batch)  # (1, action_dim)

            # --- Postprocess & send ---
            action_postprocessed = postprocessor(action).squeeze(0).cpu()  # (action_dim,)
            action_dict = {
                key: action_postprocessed[i].item()
                for i, key in enumerate(robot.action_features)
            }
            action_processed = robot_action_processor((action_dict, None))
            robot.send_action(action_processed)

            step += 1
            if step % 50 == 0:
                elapsed = time.time() - start_time
                logger.info(f"[MAIN] step={step}  elapsed={elapsed:.1f}s  task='{active_task}'")

            # Pace the loop to target fps
            dt = time.perf_counter() - t0
            time.sleep(max(0.0, action_interval - dt - 0.001))

    finally:
        logger.info("Shutting down")
        shutdown_event.set()
        voice_shutdown.set()
        voice_proc.join(timeout=5)
        if voice_proc.is_alive():
            voice_proc.terminate()
        robot.disconnect()
        logger.info(f"Robot disconnected — total steps: {step}")


if __name__ == "__main__":
    main()
