#!/usr/bin/env python3
"""
whispr_flow/dictate.py

DIY Wispr Flow for Mac.
- Double-tap Fn key  → start recording
- Single-tap Fn key  → stop recording, transcribe, paste at cursor

Requirements (install once):
    pip install faster-whisper sounddevice pyperclip pynput

On first run you'll also need to grant Accessibility + Microphone permissions
to your terminal app in System Settings > Privacy & Security.
"""

import time
import threading
import tempfile
import os
import sys
import wave

import sounddevice as sd
import numpy as np
import subprocess
import pyperclip
from pynput import keyboard

# ─────────────────────────────────────────────
# CONFIG — tweak to taste
# ─────────────────────────────────────────────
SAMPLE_RATE      = 16_000          # Whisper expects 16 kHz
CHANNELS         = 1
DOUBLE_TAP_MS    = 400             # Max ms between two Fn taps to count as double
MODEL_SIZE       = "base.en"       # tiny.en / base.en / small.en / medium.en / large-v3
                                   # larger = more accurate but slower first load
DEVICE           = "cpu"           # "cpu" or "cuda" if you have an NVIDIA GPU
COMPUTE_TYPE     = "int8"          # int8 is fastest on CPU; float16 on GPU
PASTE_DELAY      = 0.15            # seconds to wait before CMD+V

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
recording        = False
audio_frames     = []
fn_tap_times     = []              # timestamps of recent Fn key presses
model            = None
model_lock       = threading.Lock()
STREAM           = None

# ─────────────────────────────────────────────
# LAZY-LOAD MODEL (happens once in background)
# ─────────────────────────────────────────────
def load_model():
    global model
    try:
        from faster_whisper import WhisperModel
        print(f"[whispr] Loading Whisper model '{MODEL_SIZE}'... (first run takes ~30s to download)")
        m = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        with model_lock:
            model = m
        print("[whispr] Model ready. Double-tap Fn to start recording.")
    except ImportError:
        print("[ERROR] faster-whisper not installed. Run: pip install faster-whisper")
        sys.exit(1)


# ─────────────────────────────────────────────
# AUDIO RECORDING
# ─────────────────────────────────────────────
def audio_callback(indata, frames, time_info, status):
    if recording:
        audio_frames.append(indata.copy())


def start_recording():
    global recording, audio_frames, STREAM
    if recording:
        return
    audio_frames = []
    recording = True
    STREAM = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
    )
    STREAM.start()
    print("[whispr] 🔴 Recording... (tap Fn once to stop)")


def stop_and_transcribe():
    global recording, STREAM
    if not recording:
        return
    recording = False
    if STREAM:
        STREAM.stop()
        STREAM.close()
        STREAM = None

    if not audio_frames:
        print("[whispr] No audio captured.")
        return

    print("[whispr] Transcribing...")
    threading.Thread(target=_transcribe_and_paste, daemon=True).start()


def _transcribe_and_paste():
    with model_lock:
        m = model

    if m is None:
        print("[whispr] Model not loaded yet. Try again in a moment.")
        return

    # Flatten frames → numpy array → write to temp WAV
    audio_data = np.concatenate(audio_frames, axis=0).flatten()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    try:
        with wave.open(tmp_path, "w") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)                         # 16-bit
            wf.setframerate(SAMPLE_RATE)
            pcm = (audio_data * 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())

        segments, info = m.transcribe(
            tmp_path,
            beam_size=5,
            language="en",                             # remove this line for auto-detect
            vad_filter=True,                           # strips silence
            vad_parameters=dict(min_silence_duration_ms=300),
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()

        if not text:
            print("[whispr] Nothing detected.")
            return

        print(f"[whispr] ✅ '{text}'")
        _paste_text(text)

    finally:
        os.unlink(tmp_path)


def _paste_text(text: str):
    pyperclip.copy(text)
    time.sleep(PASTE_DELAY)
    subprocess.run([
        "osascript", "-e",
        'tell application "System Events" to keystroke "v" using command down'
    ])


# ─────────────────────────────────────────────
# FN KEY DETECTION
# The Fn key on Mac comes through as Key.f20 in pynput on most
# modern Macs (Apple Silicon + Intel). If yours differs, run
# `python3 -c "from pynput import keyboard; k=keyboard.Listener(on_press=print); k.start(); k.join()"``
# and tap Fn to see what key code appears, then update FN_KEY below.
# ─────────────────────────────────────────────
FN_KEY = keyboard.Key.f20

def on_press(key):
    global fn_tap_times

    if key != FN_KEY:
        return

    now = time.time()
    fn_tap_times = [t for t in fn_tap_times if now - t < DOUBLE_TAP_MS / 1000]
    fn_tap_times.append(now)

    if len(fn_tap_times) >= 2:
        # Double-tap → start recording
        fn_tap_times = []
        if not recording:
            threading.Thread(target=start_recording, daemon=True).start()
        # If already recording, a double-tap does nothing (single tap stops)
    else:
        # Single tap while recording → stop
        if recording:
            fn_tap_times = []
            threading.Thread(target=stop_and_transcribe, daemon=True).start()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  whispr_flow — DIY voice dictation for Mac")
    print("=" * 50)
    print("  Double-tap Fn  →  start recording")
    print("  Single-tap Fn  →  stop + transcribe + paste")
    print("  Ctrl+C         →  quit")
    print("=" * 50)

    # Load model in background so startup is instant
    threading.Thread(target=load_model, daemon=True).start()

    # Start global hotkey listener
    with keyboard.Listener(on_press=on_press) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            print("\n[whispr] Bye.")
