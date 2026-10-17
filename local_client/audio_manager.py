import os
import sys
import time
import ctypes
import threading
from typing import Optional


class StarkAudioManager:
    """
    Stark OS Low-Latency Audio Manager:
    - Leverages native Windows Multi-Media (winmm.dll mciSendStringW) for 0ms CPU-efficient MP3 playback.
    - Pre-buffers audio handles in RAM so playback has zero disk-seek latency.
    - Manages simultaneous/layered playback for startup boot sounds and action trigger effects.
    """

    def __init__(self, audio_dir: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        if audio_dir is None:
            audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_files")
        self.audio_dir = audio_dir

        self.mci = None
        self._lock = threading.Lock()
        self._startup_played = False

        if sys.platform == "win32" and self.enabled:
            try:
                self.mci = ctypes.windll.winmm.mciSendStringW
            except Exception as e:
                print(f"[Audio] WinMM MCI initialization failed: {e}")
                self.mci = None

        self.aliases = {
            "startup_1": "Interface_open1.mp3",
            "startup_2": "Interface_open2.mp3",
            "action_1": "Action_sound1.mp3",
            "action_2": "Action_sound2.mp3",
        }

        self._active_aliases = []
        if self.mci:
            self._init_audio_handles()

    def _init_audio_handles(self):
        """Pre-opens all audio files with MCI aliases so they play instantly on trigger."""
        if not os.path.exists(self.audio_dir):
            print(f"[Audio] Audio directory not found: {self.audio_dir}")
            return

        with self._lock:
            for alias, filename in self.aliases.items():
                filepath = os.path.join(self.audio_dir, filename)
                if os.path.exists(filepath):
                    abs_path = os.path.abspath(filepath)
                    # Close any leftover handles with the same alias
                    self.mci(f"close {alias}", None, 0, None)
                    err = self.mci(f'open "{abs_path}" type mpegvideo alias {alias}', None, 0, None)
                    if err == 0:
                        self._active_aliases.append(alias)
                    else:
                        print(f"[Audio] Notice: Unable to pre-open {filename} (code {err})")
                else:
                    print(f"[Audio] Missing audio file: {filepath}")

    def play_startup(self):
        """Plays both interface open sound effects on application startup."""
        if not self.enabled or not self.mci or self._startup_played:
            return
        self._startup_played = True

        def _worker():
            with self._lock:
                if "startup_1" in self._active_aliases:
                    self.mci("play startup_1 from 0", None, 0, None)
                if "startup_2" in self._active_aliases:
                    self.mci("play startup_2 from 0", None, 0, None)

        threading.Thread(target=_worker, name="StarkAudio-Startup", daemon=True).start()

    def play_action(self, delay: float = 0.15):
        """
        Plays both action sound effects when an action triggers, following a short delay.
        """
        if not self.enabled or not self.mci:
            return

        def _worker():
            if delay > 0:
                time.sleep(delay)
            with self._lock:
                if "action_1" in self._active_aliases:
                    self.mci("play action_1 from 0", None, 0, None)
                if "action_2" in self._active_aliases:
                    self.mci("play action_2 from 0", None, 0, None)

        threading.Thread(target=_worker, name="StarkAudio-Action", daemon=True).start()

    def close(self):
        """Stops and releases all MCI multimedia device handles cleanly."""
        if not self.mci:
            return
        with self._lock:
            for alias in self._active_aliases:
                try:
                    self.mci(f"stop {alias}", None, 0, None)
                    self.mci(f"close {alias}", None, 0, None)
                except Exception:
                    pass
            self._active_aliases.clear()
