#!/usr/bin/env python
import os
import sys
import time
import argparse
import threading
from collections import deque
import cv2
import numpy as np

# Ensure local imports work regardless of working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
if current_dir not in sys.path:
    sys.path.append(current_dir)
if repo_root not in sys.path:
    sys.path.append(repo_root)

from dispatcher import ActionDispatcher


def draw_hud(
    frame,
    fps: float,
    stream_mode: str,
    buffer_len: int,
    max_buffer: int,
    gesture_3d: str,
    conf_3d: float,
    is_actionable_3d: bool,
    gesture_2d: str,
    conf_2d: float,
    is_actionable_2d: bool,
    threshold: float,
    cooldown_remaining: float,
    last_triggered_gesture: str,
    last_triggered_time: float,
    probabilities_3d: dict = None,
    show_debug: bool = False
):
    """Renders a sleek IronMan Heads-Up Display (HUD) on the webcam frame."""
    h, w, _ = frame.shape

    # 1. Top HUD Bar (Dark semi-transparent background)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 95), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # 2. System Status (Left Column)
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    mode_color = (0, 255, 200) if stream_mode == "dual" else ((0, 200, 255) if stream_mode == "2d" else (255, 180, 0))
    cv2.putText(frame, f"Mode: {stream_mode.upper()}", (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1, cv2.LINE_AA)
    buffer_color = (0, 255, 0) if buffer_len == max_buffer else (0, 200, 255)
    cv2.putText(frame, f"3D Buffer: [{buffer_len}/{max_buffer}]", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.50, buffer_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Threshold: {int(threshold * 100)}%", (15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    # 3. Center Section: Live Gesture Status (2D & 3D)
    center_x = w // 2 - 120

    # 3D Dynamic Gesture Line
    if stream_mode in ["dual", "3d"]:
        name_3d = gesture_3d.replace("_", " ").upper()
        if is_actionable_3d:
            color_3d = (0, 255, 0) # Green
            text_3d = f">> 3D SWIPE: {name_3d} ({conf_3d*100:.0f}%) <<"
        elif gesture_3d in ["swipe_left", "swipe_right"]:
            color_3d = (0, 220, 255) # Yellow
            text_3d = f"3D Candidate: {name_3d} ({conf_3d*100:.0f}%)"
        else:
            color_3d = (160, 160, 160)
            text_3d = f"3D Motion: {name_3d} ({conf_3d*100:.0f}%)"
        cv2.putText(frame, text_3d, (center_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_3d, 2, cv2.LINE_AA)

    # 2D Static Pose Line
    if stream_mode in ["dual", "2d"]:
        name_2d = gesture_2d.replace("_", " ").upper()
        if is_actionable_2d:
            color_2d = (0, 255, 0)
            text_2d = f">> 2D POSE: {name_2d} ({conf_2d*100:.0f}%) <<"
        elif gesture_2d != "no_gesture":
            color_2d = (0, 220, 255)
            text_2d = f"2D Candidate: {name_2d} ({conf_2d*100:.0f}%)"
        else:
            color_2d = (140, 140, 140)
            text_2d = f"2D Pose: {name_2d} ({conf_2d*100:.0f}%)"
        y_pos = 75 if stream_mode == "dual" else 50
        cv2.putText(frame, text_2d, (center_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color_2d, 2, cv2.LINE_AA)

    # 4. Cooldown Indicator (Top-Right)
    if cooldown_remaining > 0:
        cd_text = f"COOLDOWN: {cooldown_remaining:.1f}s"
        cd_color = (0, 140, 255)  # Orange
    else:
        cd_text = "READY"
        cd_color = (0, 255, 0)    # Green
    cv2.putText(frame, cd_text, (w - 180, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cd_color, 2, cv2.LINE_AA)

    # 5. Bottom Action Execution Banner (lasts 1.2s after trigger)
    time_since_trigger = time.time() - last_triggered_time
    if time_since_trigger < 1.2 and last_triggered_gesture:
        banner_overlay = frame.copy()
        cv2.rectangle(banner_overlay, (0, h - 55), (w, h), (0, 110, 0), -1)
        cv2.addWeighted(banner_overlay, 0.85, frame, 0.15, 0, frame)
        action_msg = f"HOTKEY FIRED: {last_triggered_gesture.upper()}"
        cv2.putText(frame, action_msg, (w // 2 - 170, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # 6. Optional Debug Probability Breakdown (Toggle with 'h')
    if show_debug and probabilities_3d:
        box_y = 110
        cv2.putText(frame, "3D Probabilities:", (15, box_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        for name, prob in probabilities_3d.items():
            bar_w = int(prob * 140)
            cv2.rectangle(frame, (15, box_y), (15 + bar_w, box_y + 12), (0, 200, 255), -1)
            cv2.rectangle(frame, (15, box_y), (155, box_y + 12), (70, 70, 70), 1)
            cv2.putText(frame, f"{name}: {prob*100:.0f}%", (165, box_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
            box_y += 18

    # 7. Helper Controls (Bottom Left)
    cv2.putText(frame, "[S] Stream  [T] Threshold  [C] Cooldown  [H] Debug  [Q] Quit",
                (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)


class ThreadedInferenceWorker:
    """
    Decoupled background worker that runs 2D and 3D inference asynchronously.
    Prevents OpenCV frame drops and keeps the video stream buttery smooth at 30+ FPS.
    """
    def __init__(self, engine, threshold: float = 0.70):
        self.engine = engine
        self.threshold = threshold
        self.lock = threading.Lock()
        self.running = True

        # Input request state
        self.latest_frame = None
        self.latest_buffer_snapshot = None
        self.active_mode = "dual"  # "dual", "2d", "3d"
        self.has_new_job = False

        # Output results state
        self.result_3d = {"gesture": "no_gesture", "confidence": 0.0, "is_actionable": False, "probabilities": {}}
        self.result_2d = {"gesture": "no_gesture", "confidence": 0.0, "is_actionable": False, "probabilities": {}}

        # Start daemon worker thread
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def submit_job(self, frame, buffer_snapshot, mode: str, threshold: float):
        """Called by camera loop to queue the latest frames without blocking."""
        if self.lock.acquire(blocking=False):
            try:
                self.latest_frame = frame
                self.latest_buffer_snapshot = buffer_snapshot
                self.active_mode = mode
                self.threshold = threshold
                self.has_new_job = True
            finally:
                self.lock.release()

    def get_results(self):
        """Fetches the latest prediction states."""
        with self.lock:
            return dict(self.result_3d), dict(self.result_2d)

    def _worker_loop(self):
        while self.running:
            job_frame = None
            job_buffer = None
            job_mode = "dual"
            job_thresh = self.threshold

            with self.lock:
                if self.has_new_job:
                    job_frame = self.latest_frame
                    job_buffer = self.latest_buffer_snapshot
                    job_mode = self.active_mode
                    job_thresh = self.threshold
                    self.has_new_job = False

            if job_frame is None and job_buffer is None:
                time.sleep(0.01)
                continue

            # Run 2D Static Inference (12ms)
            if job_mode in ["dual", "2d"] and job_frame is not None and self.engine.model_2d is not None:
                try:
                    res_2d = self.engine.predict_2d(job_frame, confidence_threshold=job_thresh)
                    with self.lock:
                        self.result_2d = res_2d
                except Exception as e:
                    pass

            # Run 3D Dynamic Inference (~300ms)
            # Multi-scale temporal sampling: checks 1.0s window and 1.5s window
            if job_mode in ["dual", "3d"] and job_buffer is not None and len(job_buffer) >= 30 and self.engine.model_3d is not None:
                try:
                    # 1. Fast window: 30 frames sampled every 2 (~1.0 second)
                    sampled_fast = job_buffer[-30::2][:15]
                    res_3d = self.engine.predict_3d(sampled_fast, confidence_threshold=job_thresh)

                    # 2. Relaxed window: 45 frames sampled every 3 (~1.5 seconds)
                    if len(job_buffer) >= 45:
                        sampled_slow = job_buffer[-45::3][:15]
                        res_slow = self.engine.predict_3d(sampled_slow, confidence_threshold=job_thresh)

                        # Take whichever window has higher swipe certainty
                        swipe_score_fast = max(res_3d.get("probabilities", {}).get("swipe_left", 0),
                                               res_3d.get("probabilities", {}).get("swipe_right", 0))
                        swipe_score_slow = max(res_slow.get("probabilities", {}).get("swipe_left", 0),
                                               res_slow.get("probabilities", {}).get("swipe_right", 0))

                        if swipe_score_slow > swipe_score_fast:
                            res_3d = res_slow

                    with self.lock:
                        self.result_3d = res_3d
                except Exception as e:
                    pass

            time.sleep(0.01)

    def stop(self):
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="IronMan Dual-Stream Real-Time Gesture Controller")
    parser.add_argument('--mode', type=str, choices=['dual', '2d', '3d'], default='dual',
                        help="Stream mode: 'dual' (both 2D & 3D), '2d' (poses only), '3d' (swipes only)")
    parser.add_argument('--threshold', type=float, default=0.70,
                        help="Confidence threshold for triggering shortcuts (default: 0.70)")
    parser.add_argument('--cooldown', type=float, default=1.0,
                        help="Debounce cooldown in seconds between shortcut actions (default: 1.0)")
    parser.add_argument('--camera', type=int, default=0,
                        help="Webcam camera device index (default: 0)")
    parser.add_argument('--no-actions', action='store_true', default=False,
                        help="Preview mode: detect gestures on screen without triggering OS hotkeys")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("        IRONMAN REAL-TIME DUAL-STREAM GESTURE CONTROLLER")
    print("=" * 65)
    print(f"Active Mode:          {args.mode.upper()}")
    print(f"Confidence Threshold: {args.threshold * 100:.0f}%")
    print(f"Debounce Cooldown:    {args.cooldown} seconds")
    print(f"Execute OS Actions:   {not args.no_actions}")
    print("=" * 65 + "\n")

    # 1. Initialize Inference Engine
    print("Initializing Gesture Inference Engine...")
    try:
        from cloud_backend.app.inference import GestureInferenceEngine
        engine = GestureInferenceEngine(
            default_confidence_threshold_3d=args.threshold,
            default_confidence_threshold_2d=args.threshold
        )
    except Exception as e:
        print(f"Error loading InferenceEngine: {e}")
        return

    # 2. Initialize Dispatcher
    dispatcher = ActionDispatcher(cooldown=args.cooldown)

    # 3. Initialize Webcam (use DirectShow on Windows to eliminate MSMF driver hang)
    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
    print(f"Connecting to webcam device {args.camera} via DirectShow...")
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        print(f"Error: Unable to open webcam on device index {args.camera}.")
        return
    print(f"Webcam {args.camera} connected successfully.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 4. State Tracking
    buffer_length = 45  # 45 frames at ~30 FPS = ~1.5 second motion horizon
    rolling_buffer = deque(maxlen=buffer_length)
    current_mode = args.mode
    current_threshold = args.threshold
    current_cooldown = args.cooldown

    last_triggered_gesture = ""
    last_triggered_time = 0.0
    show_debug = False

    fps = 0.0
    fps_timer = time.time()
    fps_counter = 0

    mode_cycle = ["dual", "2d", "3d"]
    threshold_levels = [0.60, 0.70, 0.75, 0.80, 0.85]
    cooldown_levels = [0.5, 0.8, 1.0, 1.5, 2.0]

    # Initialize Threaded Worker
    worker = ThreadedInferenceWorker(engine, threshold=current_threshold)

    print("\n[IronMan] Camera feed active. Gesture detection running!")
    print("Press 'q' in the window to exit, 's' to cycle stream mode (Dual / 2D / 3D).\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            fps_counter += 1
            if fps_counter >= 15:
                elapsed = time.time() - fps_timer
                fps = fps_counter / elapsed if elapsed > 0 else 30.0
                fps_timer = time.time()
                fps_counter = 0

            # Flip horizontally for natural selfie-mirror view
            frame = cv2.flip(frame, 1)

            # Append to 3D temporal buffer
            rolling_buffer.append(frame.copy())

            # Submit frames to asynchronous background worker (non-blocking)
            buffer_snap = list(rolling_buffer) if len(rolling_buffer) >= 30 else None
            worker.submit_job(
                frame=frame.copy(),
                buffer_snapshot=buffer_snap,
                mode=current_mode,
                threshold=current_threshold
            )

            # Retrieve latest predictions
            res_3d, res_2d = worker.get_results()

            # Action Dispatching: Check both streams against cooldown
            now = time.time()
            if not args.no_actions and (now - dispatcher.last_fired_time) >= dispatcher.cooldown:
                # 3D Swipe Actions take priority when actionable
                if current_mode in ["dual", "3d"] and res_3d["is_actionable"]:
                    dispatcher.execute(res_3d["gesture"])
                    last_triggered_gesture = res_3d["gesture"]
                    last_triggered_time = now
                # 2D Static Poses
                elif current_mode in ["dual", "2d"] and res_2d["is_actionable"]:
                    dispatcher.execute(res_2d["gesture"])
                    last_triggered_gesture = res_2d["gesture"]
                    last_triggered_time = now

            # Calculate remaining cooldown for HUD display
            cooldown_remaining = max(0.0, dispatcher.cooldown - (time.time() - dispatcher.last_fired_time))

            # Render Heads-Up Display
            draw_hud(
                frame=frame,
                fps=fps,
                stream_mode=current_mode,
                buffer_len=len(rolling_buffer),
                max_buffer=buffer_length,
                gesture_3d=res_3d["gesture"],
                conf_3d=res_3d["confidence"],
                is_actionable_3d=res_3d["is_actionable"],
                gesture_2d=res_2d["gesture"],
                conf_2d=res_2d["confidence"],
                is_actionable_2d=res_2d["is_actionable"],
                threshold=current_threshold,
                cooldown_remaining=cooldown_remaining,
                last_triggered_gesture=last_triggered_gesture,
                last_triggered_time=last_triggered_time,
                probabilities_3d=res_3d.get("probabilities", {}),
                show_debug=show_debug
            )

            cv2.imshow("IronMan Gesture HUD", frame)

            # Handle Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Shutting down IronMan Gesture Controller...")
                break
            elif key == ord('s'):
                # Cycle stream mode: dual -> 2d -> 3d
                cur_idx = mode_cycle.index(current_mode) if current_mode in mode_cycle else 0
                current_mode = mode_cycle[(cur_idx + 1) % len(mode_cycle)]
                print(f"[Controls] Stream mode changed to: {current_mode.upper()}")
            elif key == ord('t'):
                # Cycle threshold
                idx = (threshold_levels.index(current_threshold) + 1) % len(threshold_levels) if current_threshold in threshold_levels else 1
                current_threshold = threshold_levels[idx]
                print(f"[Controls] Threshold updated: {int(current_threshold * 100)}%")
            elif key == ord('c'):
                # Cycle cooldown
                idx = (cooldown_levels.index(current_cooldown) + 1) % len(cooldown_levels) if current_cooldown in cooldown_levels else 2
                current_cooldown = cooldown_levels[idx]
                dispatcher.cooldown = current_cooldown
                print(f"[Controls] Cooldown updated: {current_cooldown}s")
            elif key == ord('h'):
                show_debug = not show_debug

    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Hardware resources released. Goodbye, Tony.")


if __name__ == "__main__":
    main()
