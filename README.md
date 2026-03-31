# sync_inference_voice

Voice-controlled synchronous robot inference using **openwakeword** for local wake-word detection and **faster-whisper** for on-device speech-to-text. No API keys or internet connection required at runtime.

Built on top of [LeRobot](https://github.com/huggingface/lerobot) and extends the base `eval_sync.py` inference loop with a background voice listener process.

---

## How it works

```
Microphone (16 kHz mono PCM)
        │
        ▼  80 ms chunks (1280 samples)
┌───────────────────┐
│   openwakeword    │  continuously scores each chunk
│  (ONNX, local)   │  score ≥ threshold → trigger
└────────┬──────────┘
         │ wake word detected
         ▼
  print available tasks
         │
         ▼
┌───────────────────┐
│  energy-based VAD │  records until ~1.3 s silence
│  command capture  │  or 6 s hard cap
└────────┬──────────┘
         │ int16 PCM → float32 / 32768
         ▼
┌───────────────────┐
│  faster-whisper   │  local transcription (no API)
│  (ONNX, local)   │
└────────┬──────────┘
         │ transcript
         ▼
  fuzzy match → TASK_MAP
  (difflib, cutoff 0.35)
         │
         ▼
  multiprocessing.Queue
         │
         ▼
  main inference loop
  picks up new task next step
```

The voice listener runs in a **child process**, completely isolated from the real-time robot control loop. Task updates are passed via a `multiprocessing.Queue` and applied non-blocking each step.

Keyboard task switching (type `1`, `2`, `3` + Enter) remains available alongside voice.

---

## Dependencies

Managed in your Docker environment — not installed by this script:

```
openwakeword
faster-whisper
pyaudio
numpy
```

And the LeRobot framework for robot I/O and policy inference.

---

## Wake word models

openwakeword ships several pre-trained models:

| Name | Phrase |
|------|--------|
| `hey_jarvis` | "Hey Jarvis" |
| `alexa` | "Alexa" |
| `hey_mycroft` | "Hey Mycroft" |
| `hey_rhasspy` | "Hey Rhasspy" |

You can also pass a path to a custom `.onnx` or `.tflite` model file via `--wakeword_model`.

---

## Usage

```bash
python eval_sync_oww.py \
    --policy.path=<hf-repo-or-local-dir> \
    --policy.device=cuda \
    --robot.type=so101_follower \
    --robot.port=/dev/follower \
    --robot.id=videron_follower \
    --robot.cameras="{desk_cam: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, gripper: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}}" \
    --task="Put all the legos on the table in the blue bowl" \
    --robot.calibration_dir=/Experiments \
    --duration=1000 \
    --visualize=true \
    --compress_images=true \
    --wakeword_model=hey_jarvis \
    --wakeword_threshold=0.5 \
    --stt_model=small \
    --stt_device=cpu \
    --stt_compute_type=int8 \
    --mic_device_name="Logi USB Headset: Audio"
```

---

## CLI reference

### Core

| Flag | Default | Description |
|------|---------|-------------|
| `--policy.path` | *(required)* | HuggingFace repo ID or local path to the policy checkpoint |
| `--policy.device` | — | `cuda` or `cpu` |
| `--robot.type` | — | Robot type (e.g. `so101_follower`) |
| `--robot.port` | — | Serial port (e.g. `/dev/follower`) |
| `--robot.id` | — | Robot ID matching the calibration file |
| `--robot.cameras` | — | Camera config dict |
| `--task` | `""` | Starting task; overrides `TASK_MAP["1"]` at launch |
| `--duration` | `30.0` | Total run time in seconds |
| `--fps` | `30.0` | Target inference frequency |

### Visualization

| Flag | Default | Description |
|------|---------|-------------|
| `--visualize` | `false` | Enable Rerun visualization |
| `--rerun_session_name` | `eval_sync_oww` | Rerun session name |
| `--rerun_ip` | `None` | Rerun server IP |
| `--rerun_port` | `None` | Rerun server port |
| `--compress_images` | `false` | JPEG-compress images before logging to Rerun |

### Wake-word detection

| Flag | Default | Description |
|------|---------|-------------|
| `--wakeword_model` | `hey_jarvis` | Pre-trained model name or path to a custom ONNX/TFLite file |
| `--wakeword_threshold` | `0.5` | Detection confidence threshold (0–1). Lower = more sensitive, more false positives |

### Speech-to-text

| Flag | Default | Description |
|------|---------|-------------|
| `--stt_model` | `small` | faster-whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3`, `turbo` |
| `--stt_device` | `cpu` | `cpu` or `cuda`. Defaults to CPU to avoid GPU contention with the policy |
| `--stt_compute_type` | `int8` | `int8` (fast, low memory), `float16`, or `float32` |

### Microphone

| Flag | Default | Description |
|------|---------|-------------|
| `--mic_device_name` | `"Logi USB Headset: Audio"` | Case-insensitive substring matched against PyAudio input device names. Leave empty to use the system default |

---

## Task configuration

Tasks are defined in `TASK_MAP` at the top of the script:

```python
TASK_MAP = {
    "1": "Put the red lego in the blue bowl",
    "2": "Put all the legos on the table in the blue bowl",
    "3": "Move the robot to the home position",
}
```

Edit this dict to add, remove, or rename tasks. When the wake word is detected, the current task list is printed to the terminal so the operator knows what to say.

**Task matching priority:**

1. Exact numeric key match (`"1"`, `"2"`, `"3"`)
2. Fuzzy match against task descriptions (difflib, cutoff 0.35)
3. Raw transcription used verbatim as a free-form task instruction

---

## Runtime interaction

```
>>> WAKE WORD DETECTED — speak your task now <<<
Available tasks:
  [1] Put the red lego in the blue bowl
  [2] Put all the legos on the table in the blue bowl
  [3] Move the robot to the home position

[VOICE] Transcribed: 'put all the legos in the bowl' (lang=en, conf=0.99)
>>> VOICE SWITCH: Put all the legos on the table in the blue bowl
```

Keyboard shortcuts remain active at all times — type a task key and press Enter to switch immediately without speaking.

---

## Audio constants

These can be tuned at the top of the script if needed:

| Constant | Value | Description |
|----------|-------|-------------|
| `_SAMPLE_RATE` | `16000` Hz | Required by both openwakeword and Whisper |
| `_OWW_CHUNK` | `1280` samples | 80 ms — openwakeword's preferred chunk size |
| `_SPEECH_THRESHOLD` | `300` RMS | Onset level to begin recording a command |
| `_SILENCE_THRESHOLD` | `200` RMS | Level considered silence during recording |
| `_SILENCE_CHUNKS` | `20` | ~1.3 s of silence ends the utterance |
| `_RECORD_MAX_SECONDS` | `6` | Hard cap on command recording duration |
