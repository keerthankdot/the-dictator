#!/usr/bin/env python3
"""
whispr_flow/dictate.py

DIY Wispr Flow for Mac.
- Double-tap Fn  → start recording
- Single-tap Fn  → stop, transcribe, paste at cursor

Requirements:
    pip install faster-whisper sounddevice pyperclip pynput
"""

import time
import threading
import tempfile
import os
import sys
import wave
import tkinter as tk

import sounddevice as sd
import numpy as np
import subprocess
import pyperclip
from pynput import keyboard

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SAMPLE_RATE      = 16_000
CHANNELS         = 1
DOUBLE_TAP_MS    = 400
MODEL_SIZE       = "base.en"
DEVICE           = "cpu"
COMPUTE_TYPE     = "int8"
PASTE_DELAY      = 0.15

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
recording        = False
audio_frames     = []
fn_tap_times     = []
model            = None
model_lock       = threading.Lock()
STREAM           = None
target_app       = None
overlay          = None

# ─────────────────────────────────────────────
# STATUS OVERLAY — floating dot near cursor
# ─────────────────────────────────────────────
class StatusOverlay:
    _STATES = {
        'loading':      ('#888888', 0.5),
        'ready':        ('#00CC44', 0.25),
        'recording':    ('#FF3333', 0.95),
        'transcribing': ('#FF9900', 0.95),
    }
    SIZE = 20

    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.wm_attributes('-topmost', True)
        self.root.wm_attributes('-alpha', 0.85)
        self.root.wm_attributes('-transparent', True)
        self.root.configure(bg='black')

        self.canvas = tk.Canvas(
            self.root,
            width=self.SIZE,
            height=self.SIZE,
            bg='black',
            highlightthickness=0,
        )
        self.canvas.pack()
        self.dot = self.canvas.create_oval(
            2, 2, self.SIZE - 2, self.SIZE - 2,
            fill='#888888',
            outline='',
        )
        self.root.after(100, self._set_native_window_level)
        self._follow_cursor()

    def _set_native_window_level(self):
        try:
            import objc
            from AppKit import NSFloatingWindowLevel
            from AppKit import NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorStationary
            self.root.update()
            view = objc.objc_object(c_void_p=self.root.winfo_id())
            ns_win = view.window()
            ns_win.setLevel_(NSFloatingWindowLevel)
            ns_win.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces |
                NSWindowCollectionBehaviorStationary
            )
        except Exception as e:
            print(f"[whispr] Window level warning: {e}")

    def _follow_cursor(self):
        x = self.root.winfo_pointerx() + 16
        y = self.root.winfo_pointery() + 16
        self.root.geometry(f'{self.SIZE}x{self.SIZE}+{x}+{y}')
        self.root.after(40, self._follow_cursor)

    def set_status(self, status: str):
        color, alpha = self._STATES.get(status, ('#888888', 0.5))
        def _apply():
            self.canvas.itemconfig(self.dot, fill=color)
            self.root.wm_attributes('-alpha', alpha)
        self.root.after(0, _apply)

    def run(self):
        self.root.mainloop()


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def load_model():
    global model
    try:
        from faster_whisper import WhisperModel
        print(f"[whispr] Loading model '{MODEL_SIZE}'...")
        m = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        with model_lock:
            model = m
        print("[whispr] Ready. Double-tap Fn to start recording.")
        if overlay:
            overlay.set_status('ready')
    except ImportError:
        print("[ERROR] faster-whisper not installed. Run: pip install faster-whisper")
        sys.exit(1)


# ─────────────────────────────────────────────
# AUDIO
# ─────────────────────────────────────────────
def audio_callback(indata, frames, time_info, status):
    if recording:
        audio_frames.append(indata.copy())


def _get_frontmost_app():
    r = subprocess.run(
        ["osascript", "-e",
         'tell application "System Events" to get name of first application process whose frontmost is true'],
        capture_output=True, text=True
    )
    return r.stdout.strip()


def start_recording():
    global recording, audio_frames, STREAM, target_app
    if recording:
        return
    target_app = _get_frontmost_app()
    audio_frames = []
    recording = True
    STREAM = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
    )
    STREAM.start()
    print("[whispr] Recording...")
    if overlay:
        overlay.set_status('recording')


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
        if overlay:
            overlay.set_status('ready')
        return
    print("[whispr] Transcribing...")
    if overlay:
        overlay.set_status('transcribing')
    threading.Thread(target=_transcribe_and_paste, daemon=True).start()


def _transcribe_and_paste():
    with model_lock:
        m = model
    if m is None:
        print("[whispr] Model not ready yet.")
        return

    audio_data = np.concatenate(audio_frames, axis=0).flatten()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    try:
        with wave.open(tmp_path, "w") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            pcm = (audio_data * 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())

        segments, _ = m.transcribe(
            tmp_path,
            beam_size=5,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if not text:
            print("[whispr] Nothing detected.")
            return
        print(f"[whispr] '{text}'")
        _paste_text(text)
    finally:
        os.unlink(tmp_path)
        if overlay:
            overlay.set_status('ready')


def _paste_text(text: str):
    pyperclip.copy(text)
    time.sleep(PASTE_DELAY)
    if target_app:
        subprocess.run(["osascript", "-e", f'tell application "{target_app}" to activate'])
        time.sleep(0.15)
    subprocess.run([
        "osascript", "-e",
        'tell application "System Events" to keystroke "v" using command down'
    ])


# ─────────────────────────────────────────────
# FN KEY LISTENER
# ─────────────────────────────────────────────
FN_KEY = keyboard.KeyCode.from_vk(179)

def on_press(key):
    global fn_tap_times
    if key != FN_KEY:
        return
    now = time.time()
    fn_tap_times = [t for t in fn_tap_times if now - t < DOUBLE_TAP_MS / 1000]
    fn_tap_times.append(now)

    if len(fn_tap_times) >= 2:
        fn_tap_times = []
        if not recording:
            threading.Thread(target=start_recording, daemon=True).start()
    else:
        if recording:
            fn_tap_times = []
            threading.Thread(target=stop_and_transcribe, daemon=True).start()


def _run_keyboard_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("[whispr] Starting — double-tap Fn to record, single-tap to stop. Ctrl+C to quit.")
    overlay = StatusOverlay()
    threading.Thread(target=load_model, daemon=True).start()
    threading.Thread(target=_run_keyboard_listener, daemon=True).start()
    try:
        overlay.run()
    except KeyboardInterrupt:
        print("\n[whispr] Bye.")
