# CLAUDE.md — whispr_flow

DIY Wispr Flow for Mac. System-wide voice dictation via local Whisper.
No cloud. No subscription. Pastes transcribed text at the active cursor.

---

## What this project is

A single Python script (`dictate.py`) that runs as a background process.
It listens globally for the Fn key, records mic audio on double-tap,
transcribes via faster-whisper (local, offline), and pastes the result
wherever the cursor is using the system clipboard + CMD+V.

There is no UI. No server. No database. One file, one process.

---

## File structure

```
whispr_flow/
├── dictate.py      # The entire application. Everything lives here.
├── CLAUDE.md       # This file.
└── README.md       # Human setup guide (permissions, install, run).
```

---

## Architecture

```
Global key listener (pynput)
        │
        ▼
Double-tap Fn → start_recording()
        │         └── opens sd.InputStream with audio_callback
        │               └── appends indata chunks to audio_frames[]
        │
Single-tap Fn → stop_and_transcribe()
        │         └── closes InputStream
        │               └── spawns thread → _transcribe_and_paste()
        │
_transcribe_and_paste()
        │── flatten audio_frames → numpy array
        │── write to temp .wav (16kHz, mono, 16-bit PCM)
        │── faster-whisper transcribes → list of segments
        │── join segment texts → single string
        └── _paste_text() → pyperclip.copy() + pyautogui CMD+V
```

### Why threads everywhere
- `start_recording()` and `stop_and_transcribe()` are called from the pynput
  key listener thread. Audio I/O and transcription must not block that thread
  or key events get dropped. Every heavy operation spawns its own daemon thread.
- `_transcribe_and_paste()` holds `model_lock` while using the Whisper model
  to prevent concurrent transcription calls from corrupting state.

---

## Key decisions and why

### faster-whisper over openai-whisper
faster-whisper uses CTranslate2 under the hood. It is 4x faster than the
original openai-whisper on CPU and uses significantly less RAM. Same model
weights, better runtime. Always prefer faster-whisper.

### Model: base.en
Default is `base.en`. English-only models (`.en` suffix) are faster and more
accurate than multilingual models of the same size for English speech.
`base.en` hits the right speed/accuracy balance for real-time dictation on
CPU. If the user has an Apple Silicon Mac with enough RAM, `small.en` is a
meaningful accuracy upgrade with acceptable latency.

### Sample rate: 16,000 Hz — DO NOT CHANGE
Whisper was trained on 16kHz audio. Passing a different sample rate will
silently produce garbage transcriptions. sounddevice records at 16kHz,
the WAV is written at 16kHz. This must stay consistent end-to-end.

### Channels: 1 (mono) — DO NOT CHANGE
Whisper expects mono. Stereo input must be mixed down before transcription.
Recording mono directly is simpler and correct.

### Audio dtype: float32 → converted to int16 for WAV
sounddevice records as float32 in [-1.0, 1.0]. The WAV file is written as
int16 PCM (multiply by 32767). faster-whisper reads the WAV and handles
the rest. Do not change this conversion or the WAV will be unreadable.

### VAD filter: enabled
`vad_filter=True` in the transcribe call strips silence from the audio before
processing. This prevents Whisper from hallucinating text on silent recordings
(a known Whisper failure mode). Always keep this on.

### Fn key: Key.f20
On most modern Macs (Apple Silicon and Intel), pynput receives the Fn key as
`keyboard.Key.f20`. This is not guaranteed across all hardware. If the key
listener does not trigger, the user must run the key detection one-liner in
the README to find their actual key code and update `FN_KEY` in dictate.py.

### Double-tap window: 400ms
Two Fn taps within 400ms = double-tap = start recording.
A single tap while recording = stop. The `fn_tap_times` list is pruned on
every keypress to only keep taps within the window. This is stateless and
robust — no timer threads needed.

### Paste method: clipboard + CMD+V
`pyautogui.hotkey("command", "v")` is the only reliable way to paste into
arbitrary Mac apps including Electron apps, browsers, and native AppKit apps.
Direct keyboard injection via pynput fails in sandboxed apps.
The 150ms `PASTE_DELAY` before CMD+V gives the clipboard write time to settle.
Do not reduce this below ~100ms or paste will fail intermittently.

### Model loaded in background thread on startup
The Whisper model takes 1-5 seconds to load depending on size. Loading it in
a daemon thread at startup means the process is immediately ready to accept
key events. The `model_lock` ensures transcription only runs after load
completes. If the user double-taps Fn before the model is ready, they get a
clear "Model not loaded yet" message and no crash.

---

## Config knobs (top of dictate.py)

| Variable | Default | What it controls |
|---|---|---|
| `MODEL_SIZE` | `base.en` | Whisper model. See model table in README. |
| `DEVICE` | `cpu` | `cpu` or `cuda`. Use `cpu` on Mac. |
| `COMPUTE_TYPE` | `int8` | `int8` fastest on CPU. `float16` on NVIDIA GPU. |
| `DOUBLE_TAP_MS` | `400` | Max ms between Fn taps to count as double-tap. |
| `PASTE_DELAY` | `0.15` | Seconds between clipboard write and CMD+V. |
| `FN_KEY` | `Key.f20` | The pynput key object for Fn. |

---

## How to add Claude API cleanup (the planned next step)

The hook point is inside `_transcribe_and_paste()`, after the transcript is
assembled and before `_paste_text()` is called.

```python
# Current:
text = " ".join(seg.text.strip() for seg in segments).strip()
_paste_text(text)

# With Claude cleanup:
text = " ".join(seg.text.strip() for seg in segments).strip()
text = clean_transcript(text)   # add this line
_paste_text(text)
```

Add this function anywhere in the file:

```python
import anthropic

def clean_transcript(raw: str) -> str:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                "Clean up this voice transcript. Remove filler words (um, uh, like, you know). "
                "Fix grammar and punctuation. Preserve meaning exactly. "
                "Return ONLY the cleaned text, nothing else.\n\n"
                f"Transcript: {raw}"
            )
        }]
    )
    return msg.content[0].text.strip()
```

Set the API key: `export ANTHROPIC_API_KEY=sk-ant-...` in your shell profile.

Make `clean_transcript` optional via a config flag at the top of the file:

```python
USE_CLAUDE_CLEANUP = False   # set to True when API key is available
```

---

## Known failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Fn key does nothing | Wrong key code for this Mac | Run key detection one-liner from README, update `FN_KEY` |
| Fn key works but no audio | Microphone permission not granted | System Settings → Privacy → Microphone → enable terminal |
| Paste doesn't land in app | Accessibility permission not granted | System Settings → Privacy → Accessibility → enable terminal |
| Paste lands but offset | `PASTE_DELAY` too short | Increase to 0.25 |
| Hallucinated text on silence | VAD filter off | Ensure `vad_filter=True` in transcribe call |
| Garbled transcription | Sample rate mismatch | Confirm `SAMPLE_RATE = 16_000` everywhere |
| Slow first transcription | Model loading | Normal. Subsequent calls are fast. |
| Process crashes on Fn tap before model loads | Race condition | Fixed: `model_lock` + None check in `_transcribe_and_paste` |

---

## What not to break

- `SAMPLE_RATE = 16_000` — non-negotiable, Whisper requirement
- `vad_filter=True` — removing this causes hallucinations on silence
- Thread spawning for `start_recording` and `stop_and_transcribe` — blocking the pynput thread drops key events
- The `model_lock` — removing it risks concurrent transcription calls on the same model instance
- `pyperclip.copy()` before `pyautogui.hotkey()` — order matters, clipboard must be written first
