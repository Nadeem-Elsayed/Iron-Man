import os
import sys
import types
import time
import threading
import argparse
import ctypes
import requests
import cv2

# Stub matplotlib if not loaded to prevent PyInstaller import failure in MediaPipe
if "matplotlib" not in sys.modules:
    try:
        import matplotlib
    except ImportError:
        dummy_mpl = types.ModuleType("matplotlib")
        dummy_mpl.pyplot = types.ModuleType("matplotlib.pyplot")
        sys.modules["matplotlib"] = dummy_mpl
        sys.modules["matplotlib.pyplot"] = dummy_mpl.pyplot

# Ensure local_client is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dispatcher import ActionDispatcher
from hud_renderer import IronManHUD
from direction_resolver import DirectionResolver

# Optional local high-performance DirectML GPU inference engine
try:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sys.path.insert(0, os.path.join(repo_root, "cloud_backend", "app"))
    from inference import GestureInferenceEngine
    HAS_LOCAL_ENGINE = True
except Exception:
    HAS_LOCAL_ENGINE = False


class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


# Per-gesture confidence activation thresholds
GESTURE_THRESHOLDS = {
    "xsign": 0.70,        # Crossed arms/wrists: 0.70 ensures responsive, effortless tab closing
    "thumb_index2": 0.75, # Clean desktop toggle
    "heart": 0.75,
    "hand_heart": 0.75,
    "mute": 0.75,
    "take_photo": 0.75,
    "take_picture": 0.75,
    "grabbing": 0.75,
    "stop": 0.75
}


class ThreadedCloudWorker:
    """
    Decoupled background inference worker:
    1. Direct local DirectML GPU inference (<2ms) when 9-class weights exist.
    2. HTTP cloud microservice fallback when configured or running standalone.
    3. Runs asynchronously in a daemon thread so the camera render loop maintains 30-60 FPS without stutter.
    """
    def __init__(self, url: str, threshold: float, dispatcher: ActionDispatcher, actions_enabled: bool = True):
        self.url = url
        self.threshold = threshold
        self.dispatcher = dispatcher
        self.actions_enabled = actions_enabled

        self.lock = threading.Lock()
        self.latest_frame = None
        self.has_new_frame = False
        self.running = True

        self.local_engine = None
        # Auto-detect local weights if url is default or "local"
        is_default_cloud = ("lambda-url" in url) or (url == "local")
        if HAS_LOCAL_ENGINE and is_default_cloud:
            try:
                print("[IronMan Engine] Initializing local 9-class DirectML GPU inference engine...")
                self.local_engine = GestureInferenceEngine()
                if self.local_engine.model_2d is not None:
                    print(f"[IronMan Engine] Local Engine ACTIVE on {self.local_engine.device}! (9 Classes: {self.local_engine.LABELS_2D})")
                else:
                    self.local_engine = None
            except Exception as e:
                print(f"[IronMan Engine] Local engine initialization notice: {e}")
                self.local_engine = None

        self.session = requests.Session() if self.local_engine is None else None

        # Temporal consensus counters (filters out transient motion between gestures)
        self.streak_gesture = None
        self.streak_count = 0
        self.REQUIRED_STREAKS = {
            "xsign": 2,          # 2 frames (~60ms) - swift tab closing
            "grabbing": 5,       # 5 frames (~160ms) - deliberate held fist, prevents motion trigger
            "mute": 4,           # 4 frames (~130ms) - deliberate hold, prevents false trigger when crossing arms
            "thumb_index2": 3,   # 3 frames (~100ms) - deliberate desktop toggle
            "hand_heart": 3,
            "heart": 3,
            "take_photo": 3,
            "take_picture": 3,
            "stop": 4
        }

        self.latest_result = {
            "gesture": "no_gesture",
            "confidence": 0.0,
            "is_actionable": False,
            "latency_ms": 0.0,
            "status": "Online (Local GPU)" if self.local_engine else "Connecting..."
        }

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def submit_frame(self, frame, threshold: float, actions_enabled: bool, num_hands: int = 1):
        """Non-blocking submission of the latest frame with multi-hand context."""
        if self.lock.acquire(blocking=False):
            try:
                self.latest_frame = frame
                self.threshold = threshold
                self.actions_enabled = actions_enabled
                self.num_hands = num_hands
                self.has_new_frame = True
            finally:
                self.lock.release()

    def get_result(self):
        """Fetches the latest prediction state atomically and consumes single-frame event flags."""
        with self.lock:
            res = dict(self.latest_result)
            self.latest_result["just_fired"] = False
            return res

    def stop(self):
        self.running = False
        if self.session:
            self.session.close()

    def _worker_loop(self):
        while self.running:
            job_frame = None
            job_thresh = self.threshold
            job_actions = self.actions_enabled
            job_hands = 1

            with self.lock:
                if self.has_new_frame:
                    job_frame = self.latest_frame
                    job_thresh = self.threshold
                    job_actions = self.actions_enabled
                    job_hands = getattr(self, "num_hands", 1)
                    self.has_new_frame = False

            if job_frame is None:
                time.sleep(0.005)
                continue

            try:
                if self.local_engine:
                    t0 = time.perf_counter()
                    # Query engine with 0.50 floor to maintain rich real-time confidence telemetry
                    res = self.local_engine.predict_2d(job_frame, confidence_threshold=0.50)
                    rtt_ms = (time.perf_counter() - t0) * 1000.0

                    gesture = res.get("gesture", "no_gesture")
                    confidence = res.get("confidence", 0.0)
                    target_thresh = GESTURE_THRESHOLDS.get(gesture, job_thresh)
                    is_actionable = (gesture != "no_gesture") and (confidence >= target_thresh)

                    # Two-handed context guard: Mute, Point, and Grabbing are strictly single-handed gestures.
                    # If two hands are visible (e.g. crossing arms or finger guns), suppress mute, point, and grabbing.
                    if job_hands >= 2 and gesture in ("mute", "point", "grabbing"):
                        is_actionable = False

                    # Temporal consensus filter: deliberate hold confirmation
                    if is_actionable:
                        if gesture == self.streak_gesture:
                            self.streak_count += 1
                        else:
                            self.streak_gesture = gesture
                            self.streak_count = 1
                    else:
                        self.streak_count = 0
                        self.streak_gesture = None

                    req_streak = self.REQUIRED_STREAKS.get(gesture, 3)
                    gesture_confirmed = is_actionable and (self.streak_count >= req_streak)

                    just_fired = False
                    if gesture_confirmed and job_actions:
                        just_fired = self.dispatcher.execute(gesture)
                        if just_fired:
                            print(f"[IronMan Action] >>> MODEL TRIGGERED: '{gesture}' ({confidence*100:.1f}%) [Held: {self.streak_count} frames] <<<")
                            self.streak_count = 0  # Reset streak after firing to prevent runaway repeat triggers

                    with self.lock:
                        self.latest_result = {
                            "gesture": gesture,
                            "confidence": confidence,
                            "target_thresh": target_thresh,
                            "is_actionable": gesture_confirmed,
                            "just_fired": just_fired,
                            "latency_ms": rtt_ms,
                            "status": f"Local GPU ({rtt_ms:.1f}ms)"
                        }
                else:
                    # Downsample to 320x320 for lightning-fast network transport (ResNet normalizes to 224x224 anyway)
                    resized = cv2.resize(job_frame, (320, 320), interpolation=cv2.INTER_AREA)
                    _, encoded = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    files = {'file': ('frame.jpg', encoded.tobytes(), 'image/jpeg')}

                    t0 = time.perf_counter()
                    resp = self.session.post(
                        f"{self.url}?threshold={job_thresh}",
                        files=files,
                        timeout=0.5
                    )
                    rtt_ms = (time.perf_counter() - t0) * 1000.0

                    if resp.status_code == 200:
                        data = resp.json()
                        gesture = data.get("gesture", "no_gesture")
                        confidence = data.get("confidence", 0.0)
                        is_actionable = data.get("is_actionable", False)

                        just_fired = False
                        if is_actionable and job_actions:
                            just_fired = self.dispatcher.execute(gesture)
                            if just_fired:
                                print(f"[IronMan Action] >>> MODEL TRIGGERED: '{gesture}' ({confidence*100:.1f}%) <<<")

                        with self.lock:
                            self.latest_result = {
                                "gesture": gesture,
                                "confidence": confidence,
                                "is_actionable": is_actionable,
                                "just_fired": just_fired,
                                "latency_ms": rtt_ms,
                                "status": "Online"
                            }
                    else:
                        with self.lock:
                            self.latest_result["status"] = f"HTTP {resp.status_code}"

            except requests.exceptions.RequestException:
                with self.lock:
                    self.latest_result["status"] = "Offline / Retrying"
            except Exception as e:
                with self.lock:
                    self.latest_result["status"] = f"Notice: {e}"

            time.sleep(0.005)


def main():
    parser = argparse.ArgumentParser(description="IronMan High-FPS Cloud Webcam Client")
    parser.add_argument("--url", type=str, default=os.getenv("IRONMAN_API_URL", "https://nxebhxb63n4xfilodiwddn4jgq0rrcij.lambda-url.us-east-2.on.aws/predict"),
                        help="FastAPI microservice endpoint (default: AWS Lambda production URL or IRONMAN_API_URL env var)")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Confidence threshold to trigger OS action (default: 0.75)")
    parser.add_argument("--cooldown", type=float, default=1.2,
                        help="Debounce cooldown seconds between actions (default: 1.2)")
    parser.add_argument("--no-actions", action="store_true", default=False,
                        help="Run in observation mode without firing keystrokes")
    parser.add_argument("--no-pin", action="store_true", default=False,
                        help="Disable Always-on-Top top-right Picture-in-Picture mode")
    parser.add_argument("--hide-hints", action="store_true", default=False,
                        help="Hide the shortcut hint legend in the bottom left")
    parser.add_argument("--no-sound", action="store_true", default=False,
                        help="Disable Stark OS audio sound effects")
    parser.add_argument("--startup-volume", type=int, default=300,
                        help="Startup sound effect volume (0-1000, default: 300)")
    parser.add_argument("--warmup-seconds", type=float, default=2.0,
                        help="Startup grace period in seconds before gesture actions can fire (default: 2.0)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam device index (default: 0)")
    args = parser.parse_args()

    # Initialize the webcam with DirectShow backend for instant startup on Windows
    backend = cv2.CAP_DSHOW if cv2.__file__ and hasattr(cv2, 'CAP_DSHOW') else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        print(f"Error: Unable to open camera {args.camera}")
        return

    # Set explicit Windows AppUserModelID so taskbar groups under this standalone IronMan application
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("starkindustries.ironman.gesturehud.1.0")
    except Exception:
        pass

    # Setup Picture-in-Picture window pinned to the top-right of your screen
    window_name = "STARK OS // IronMan Gesture HUD"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.waitKey(25)  # Pump Win32 message queue so the HWND is instantiated

    is_pinned = not args.no_pin
    hwnd = ctypes.windll.user32.FindWindowW(None, window_name)

    import pyautogui
    screen_w, screen_h = pyautogui.size()
    win_w, win_h = 480, 270
    pos_x = max(0, screen_w - win_w - 20)
    pos_y = 30

    # Initialize HUD Renderer & attach Iron Man Icon to window and taskbar
    hud = IronManHUD(
        target_size=(win_w, win_h),
        enable_audio=not args.no_sound,
        startup_volume=args.startup_volume
    )
    if hwnd:
        hud.set_window_icon(hwnd)

    if is_pinned and hwnd:
        try:
            # Course 1: True Floating Picture-in-Picture Tool Overlay
            # 1. Remove Minimize & Maximize buttons from window style to eliminate DWM freeze
            GWL_STYLE = -16
            WS_MINIMIZEBOX = 0x00020000
            WS_MAXIMIZEBOX = 0x00010000
            cur_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, cur_style & ~WS_MINIMIZEBOX & ~WS_MAXIMIZEBOX)

            # 2. Apply WS_EX_TOOLWINDOW (0x00000080) and WS_EX_TOPMOST (0x00000008):
            #    - Eliminates taskbar desync and removes window from Win+D desktop minimizing
            #    - Maintains pure topmost floating HUD without stealing OS window focus
            GWL_EXSTYLE = -20
            WS_EX_TOOLWINDOW = 0x00000080
            WS_EX_TOPMOST = 0x00000008
            cur_ex = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, (cur_ex | WS_EX_TOOLWINDOW | WS_EX_TOPMOST) & ~0x08000000)

            # 3. Position and update window frame
            cv2.resizeWindow(window_name, win_w, win_h)
            cv2.moveWindow(window_name, pos_x, pos_y)
            # SWP_FRAMECHANGED (0x0020) | SWP_SHOWWINDOW (0x0040) | SWP_NOACTIVATE (0x0010)
            ctypes.windll.user32.SetWindowPos(hwnd, -1, pos_x, pos_y, win_w, win_h, 0x0020 | 0x0040 | 0x0010)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception as e:
            print(f"Window positioning notice: {e}")

    # Initialize the dispatcher mapping gestures to OS commands
    dispatcher = ActionDispatcher(cooldown=args.cooldown)
    direction_resolver = DirectionResolver()
    actions_enabled = not args.no_actions
    show_hints = not args.hide_hints

    # Start threaded inference worker
    worker = ThreadedCloudWorker(
        url=args.url,
        threshold=args.threshold,
        dispatcher=dispatcher,
        actions_enabled=actions_enabled
    )

    print("=" * 65)
    print("  STARK OS // IronMan High-FPS Cloud Gesture HUD Online")
    print(f"  Target Endpoint: {args.url}")
    print(f"  Confidence Threshold: {args.threshold:.2f}")
    print(f"  Always-on-Top Pinned: {'YES (Top-Right HUD)' if is_pinned else 'NO'}")
    print(f"  Startup Grace Period: {args.warmup_seconds:.1f}s (Disarmed during boot)")
    print(f"  OS Actions: {'ENABLED' if actions_enabled else 'DISABLED (Observation Mode)'}")
    print(f"  Startup Audio: {'ENABLED (Quiet)' if not args.no_sound else 'DISABLED'}")
    print("  Controls: [A] Actions  [P] Pin  [H] Toggle Hints  [T] Threshold  [Q] Quit")
    print("  Emergency Exit: [Ctrl + Alt + Q] (Global killswitch from ANY window)")
    print("=" * 65)

    app_start_time = time.time()
    warmup_announced = False
    last_time = app_start_time
    last_fg_window = None
    last_hud_gesture = "no_gesture"
    last_hud_trigger_time = 0.0
    last_diagnostic_time = 0.0

    try:
        while True:
            # Detect if user closed window with 'X' button
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\n[IronMan] Window closed by user. Exiting cleanly...")
                break

            # Global Emergency Killswitch (Ctrl + Alt + Q)
            try:
                # 0x11 = VK_CONTROL, 0x12 = VK_MENU (Alt), 0x51 = ord('Q')
                if (ctypes.windll.user32.GetAsyncKeyState(0x11) & 0x8000) and \
                   (ctypes.windll.user32.GetAsyncKeyState(0x12) & 0x8000) and \
                   (ctypes.windll.user32.GetAsyncKeyState(0x51) & 0x8000):
                    print("\n[IronMan] Emergency killswitch triggered (Ctrl + Alt + Q). Exiting...")
                    break
            except Exception:
                pass

            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame from hardware.")
                break

            now = time.time()
            last_time = now

            # Check startup warmup grace period (actions disarmed during first N seconds)
            is_warmed_up = (now - app_start_time) >= args.warmup_seconds
            if is_warmed_up and not warmup_announced:
                warmup_announced = True
                print(f"--> [IronMan] Startup warmup complete ({args.warmup_seconds:.1f}s). OS Gesture Actions are now ARMED.")

            # Flip horizontally for natural mirror view
            frame = cv2.flip(frame, 1)

            # Local Real-Time Direction & Posture Analysis (Modal Alt-Tab & Pointing Vectors)
            dir_res = direction_resolver.analyze_frame(frame)
            if actions_enabled and is_warmed_up:
                local_action = None
                # Only evaluate single-handed Alt-Tab trigger/navigation if exactly 1 hand is present
                if dir_res.get("num_hands") == 1:
                    if dir_res.get("is_grabbing"):
                        local_action = "grabbing"
                    elif dir_res.get("is_stop"):
                        local_action = "stop"
                    elif dir_res.get("is_pointing"):
                        d = dir_res.get("direction")
                        if d in ("point_right", "point_left"):
                            local_action = d

                if local_action:
                    fired = dispatcher.execute(local_action)
                    if fired:
                        print(f"[IronMan Action] >>> RESOLVER TRIGGERED: '{local_action}' <<<")
                        hud.trigger_action_hud(local_action)
                        last_hud_gesture = local_action
                        last_hud_trigger_time = now

            # Submit latest frame to background inference worker (actions strictly disarmed during warmup)
            worker.submit_frame(frame, args.threshold, actions_enabled and is_warmed_up, num_hands=dir_res.get("num_hands", 1))

            # Check dynamic window client size in case user resized it
            if hwnd:
                rect = RECT()
                if ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect)):
                    cw = rect.right - rect.left
                    ch = rect.bottom - rect.top
                    if cw > 150 and ch > 100:
                        win_w, win_h = cw, ch

            # Read latest inference result
            res = worker.get_result()
            gesture = res.get("gesture", "no_gesture")
            confidence = res.get("confidence", 0.0)
            target_thresh = res.get("target_thresh", GESTURE_THRESHOLDS.get(gesture, args.threshold))
            just_fired = res.get("just_fired", False)

            # Trigger HUD dials on gestures recognized with target confidence (armed after warmup)
            if is_warmed_up and gesture != "no_gesture":
                # Outside Alt-Tab mode, point is not an active action, so don't flood the HUD or console
                if gesture == "point" and not dispatcher.alt_tab_active:
                    pass
                elif confidence >= target_thresh:
                    if (gesture != last_hud_gesture) or (now - last_hud_trigger_time > 1.2):
                        hud.trigger_action_hud(gesture)
                        last_hud_gesture = gesture
                        last_hud_trigger_time = now
                        action_flag = "ACTION EXECUTED" if just_fired else "Detected"
                        print(f"[IronMan HUD] >>> CONFIRMED '{gesture}' ({confidence*100:.1f}%) -> {action_flag} <<<")
                elif confidence >= 0.50:
                    # Informative real-time telemetry if a gesture is detected slightly below the action threshold
                    if (now - last_diagnostic_time > 1.0) and (gesture != "point"):
                        last_diagnostic_time = now
                        print(f"[IronMan HUD] Detecting '{gesture}' ({confidence*100:.1f}%) [Threshold: {target_thresh*100:.0f}% - hold pose clearly in frame]")
            elif gesture == "no_gesture":
                last_hud_gesture = "no_gesture"

            # Resize display frame to native window size for crisp pixel-perfect UI
            display_frame = cv2.resize(frame, (win_w, win_h), interpolation=cv2.INTER_LINEAR)

            # Render minimalist Stark OS HUD (Zero text clutter, optional hints only)
            hud_frame = hud.render(frame=display_frame, show_hints=show_hints)

            cv2.imshow(window_name, hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                actions_enabled = not actions_enabled
                print(f"--> OS Actions toggled to: {'ENABLED' if actions_enabled else 'DISABLED'}")
            elif key == ord('p'):
                is_pinned = not is_pinned
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1 if is_pinned else 0)
                if is_pinned and hwnd:
                    ctypes.windll.user32.SetWindowPos(hwnd, -1, pos_x, pos_y, win_w, win_h, 0x0040 | 0x0010)
                elif not is_pinned and hwnd:
                    ctypes.windll.user32.SetWindowPos(hwnd, -2, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)  # HWND_NOTOPMOST
                print(f"--> Always-On-Top Pinned: {'YES' if is_pinned else 'NO'}")
            elif key == ord('h'):
                show_hints = not show_hints
                print(f"--> Shortcut Hints: {'SHOWN' if show_hints else 'HIDDEN'}")
            elif key == ord('t'):
                # Cycle threshold 0.70 -> 0.80 -> 0.85 -> 0.90 -> 0.95 -> 0.70
                thresholds = [0.70, 0.80, 0.85, 0.90, 0.95]
                curr_idx = thresholds.index(args.threshold) if args.threshold in thresholds else 2
                args.threshold = thresholds[(curr_idx + 1) % len(thresholds)]
                print(f"--> Threshold set to: {args.threshold:.2f}")

    finally:
        direction_resolver.close()
        dispatcher.cleanup()
        hud.close()
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()