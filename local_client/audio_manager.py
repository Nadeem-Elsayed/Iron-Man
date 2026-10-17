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
    
    IMPORTANT WIN32 ARCHITECTURAL NOTE:
    In Windows WinMM (winmm.dll), MCI device aliases are bound to the specific OS thread
    that opened them. If an alias is opened on Thread A, attempting to call 'play' on
    Thread B results in MCI error 263 ('The specified device is not open or is not recognized').
    
    By keeping all 'open', 'play', 'setaudio', and 'close' calls strictly inside this single
    persistent worker thread, audio playback is 100% reliable, zero-latency, and immune to
    thread-termination cutoff.
    """

    def __init__(self, audio_dir: str):
        super().__init__(name="StarkAudioWorker", daemon=True)
        self.audio_dir = audio_dir
        self.cmd_queue = queue.Queue()
        self.mci = None
        self.running = True
        self.active_aliases: List[str] = []

        if sys.platform == "win32":
            try:
                self.mci = ctypes.windll.winmm.mciSendStringW
            except Exception as e:
                print(f"[Audio] WinMM MCI initialization failed: {e}")

        self.alias_map: Dict[str, str] = {
            "startup_1": "Interface_open1.mp3",
            "startup_2": "Interface_open2.mp3",
            "action_1": "Action_sound1.mp3",
            "action_2": "Action_sound2.mp3",
        }

    def run(self):
        if not self.mci or not os.path.exists(self.audio_dir):
            return

        # Pre-open all audio handles on THIS dedicated thread
        for alias, filename in self.alias_map.items():
            filepath = os.path.join(self.audio_dir, filename)
            if os.path.exists(filepath):
                abs_path = os.path.abspath(filepath)
                self.mci(f"close {alias}", None, 0, None)
                err = self.mci(f'open "{abs_path}" type mpegvideo alias {alias}', None, 0, None)
                if err == 0:
                    self.active_aliases.append(alias)
                    # Maximize volume (MCI scale 0 to 1000)
                    self.mci(f"setaudio {alias} volume to 1000", None, 0, None)
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

            if cmd == "play":
                aliases, delay, log_msg = payload
                if delay > 0:
                    time.sleep(delay)
                for alias in aliases:
                    if alias in self.active_aliases:
                        self.mci(f"play {alias} from 0", None, 0, None)
                if log_msg:
                    print(log_msg)

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
    Thread-safe, non-blocking, and handles startup and action SFX sequencing.
    """

    def __init__(self, audio_dir: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        if audio_dir is None:
            audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_files")
        self.audio_dir = audio_dir

        self.worker: Optional[StarkAudioWorker] = None
        self._startup_played = False

        if self.enabled and sys.platform == "win32":
            self.worker = StarkAudioWorker(self.audio_dir)
            self.worker.start()

    def play_startup(self):
        """Triggers both interface open sound effects concurrently on startup."""
        if not self.enabled or not self.worker or self._startup_played:
            return
        self._startup_played = True
        self.worker.cmd_queue.put((
            "play",
            (["startup_1", "startup_2"], 0.0, "[Stark Audio] >> Startup interface sound effects engaged.")
        ))

    def play_action(self, delay: float = 0.15, gesture: str = ""):
        """
        Triggers both action sound effects after a short delay upon gesture recognition.
        """
        if not self.enabled or not self.worker:
            return
        log_msg = f"[Stark Audio] >> Action sound effects triggered ({gesture}) [delay: {delay:.2f}s]" if gesture else ""
        self.worker.cmd_queue.put((
            "play",
            (["action_1", "action_2"], max(0.0, delay), log_msg)
        ))

    def close(self):
        """Cleanly releases all audio handles and terminates the worker."""
        if self.worker and self.worker.is_alive():
            self.worker.cmd_queue.put(("close", None))
