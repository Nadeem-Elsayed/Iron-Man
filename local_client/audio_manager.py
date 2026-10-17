import os
import sys
import time
import queue
import ctypes
import threading
from typing import Optional, List, Dict


class StarkAudioWorker(threading.Thread):
    """
    Dedicated background audio worker thread.
    Manages WinMM MCI device aliases strictly on its own thread context to ensure 
    100% reliable, zero-latency playback without thread cutoff.
    """

    def __init__(self, audio_dir: str, volume: int = 300):
        super().__init__(name="StarkAudioWorker", daemon=True)
        self.audio_dir = audio_dir
        self.volume = max(0, min(1000, volume))  # MCI volume scale: 0 to 1000
        self.cmd_queue = queue.Queue()
        self.mci = None
        self.running = True
        self.active_aliases: List[str] = []

        if sys.platform == "win32":
            try:
                self.mci = ctypes.windll.winmm.mciSendStringW
            except Exception as e:
                print(f"[Audio] WinMM MCI initialization failed: {e}")

        # Only startup interface sound effects
        self.alias_map: Dict[str, str] = {
            "startup_1": "Interface_open1.mp3",
            "startup_2": "Interface_open2.mp3",
        }

    def run(self):
        if not self.mci or not os.path.exists(self.audio_dir):
            return

        # Pre-open startup audio handles on THIS dedicated thread
        for alias, filename in self.alias_map.items():
            filepath = os.path.join(self.audio_dir, filename)
            if os.path.exists(filepath):
                abs_path = os.path.abspath(filepath)
                self.mci(f"close {alias}", None, 0, None)
                err = self.mci(f'open "{abs_path}" type mpegvideo alias {alias}', None, 0, None)
                if err == 0:
                    self.active_aliases.append(alias)
                    # Set quieter volume (default 300 / 1000 = 30%)
                    self.mci(f"setaudio {alias} volume to {self.volume}", None, 0, None)
                else:
                    print(f"[Audio] Notice: Unable to pre-open {filename} (MCI code {err})")
            else:
                print(f"[Audio] Notice: Audio file not found: {filepath}")

        # Command processing loop
        while self.running:
            try:
                task = self.cmd_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            cmd, payload = task

            if cmd == "play_startup":
                for alias in ["startup_1", "startup_2"]:
                    if alias in self.active_aliases:
                        self.mci(f"play {alias} from 0", None, 0, None)
                print("[Stark Audio] >> Startup interface sound effects engaged (volume: 30%).")

            elif cmd == "close":
                for alias in self.active_aliases:
                    try:
                        self.mci(f"stop {alias}", None, 0, None)
                        self.mci(f"close {alias}", None, 0, None)
                    except Exception:
                        pass
                self.active_aliases.clear()
                self.running = False


class StarkAudioManager:
    """
    Public facade for Stark OS audio subsystem.
    Handles startup boot audio at a pleasant, non-intrusive volume.
    """

    def __init__(self, audio_dir: Optional[str] = None, enabled: bool = True, volume: int = 300):
        self.enabled = enabled
        if audio_dir is None:
            if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                audio_dir = os.path.join(sys._MEIPASS, "audio_files")
            else:
                audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_files")
        self.audio_dir = audio_dir
        self.volume = volume

        self.worker: Optional[StarkAudioWorker] = None
        self._startup_played = False

        if self.enabled and sys.platform == "win32":
            self.worker = StarkAudioWorker(self.audio_dir, volume=self.volume)
            self.worker.start()

    def play_startup(self):
        """Triggers both interface open sound effects concurrently on startup."""
        if not self.enabled or not self.worker or self._startup_played:
            return
        self._startup_played = True
        self.worker.cmd_queue.put(("play_startup", None))

    def close(self):
        """Cleanly releases all audio handles and terminates the worker."""
        if self.worker and self.worker.is_alive():
            self.worker.cmd_queue.put(("close", None))
