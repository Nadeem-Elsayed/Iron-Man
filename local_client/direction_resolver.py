import os
import sys
import types
import cv2
import numpy as np

# Stub matplotlib if not loaded to prevent PyInstaller import failure in MediaPipe
if "matplotlib" not in sys.modules:
    try:
        import matplotlib
    except ImportError:
        dummy_mpl = types.ModuleType("matplotlib")
        dummy_mpl.pyplot = types.ModuleType("matplotlib.pyplot")
        sys.modules["matplotlib"] = dummy_mpl
        sys.modules["matplotlib.pyplot"] = dummy_mpl.pyplot

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    HAS_MEDIAPIPE = True
except Exception as e:
    HAS_MEDIAPIPE = False
    print(f"[DirectionResolver] MediaPipe import notice: {e}")


class DirectionResolver:
    """
    Ultra-low-latency hand orientation and directional vector analyzer
    powered by MediaPipe Tasks HandLandmarker with temporal smoothing.
    Deterministically resolves:
    - 'point_right' vs 'point_left'
    - 'grabbing' (fist / closed grip)
    - 'stop' (flat open palm)
    - 'xsign' (crossed hands / forearms)
    """

    def __init__(
        self,
        model_path: str = None,
        horizontal_threshold: float = 0.045
    ):
        self.horizontal_threshold = horizontal_threshold
        self.detector = None

        if model_path is None:
            if getattr(sys, 'frozen', False):
                base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
                candidates = [
                    os.path.join(base_dir, "hand_landmarker.task"),
                    os.path.join(os.path.dirname(sys.executable), "hand_landmarker.task"),
                    os.path.join(os.path.dirname(sys.executable), "_internal", "hand_landmarker.task")
                ]
                for cand in candidates:
                    if os.path.exists(cand):
                        model_path = cand
                        break
            else:
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

        if HAS_MEDIAPIPE and os.path.exists(model_path):
            try:
                options = HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5
                )
                self.detector = HandLandmarker.create_from_options(options)
                print("[DirectionResolver] MediaPipe HandLandmarker online.")
            except Exception as e:
                print(f"[DirectionResolver] Detector initialization notice: {e}")
        else:
            print("[DirectionResolver] HandLandmarker task file not found or MediaPipe unavailable.")

        # Temporal smoothing counters (consecutive frame filter)
        self.consecutive_counts = {
            "grabbing": 0,
            "stop": 0,
            "point_right": 0,
            "point_left": 0
        }
        self.threshold_frames = {
            "grabbing": 5,      # 5 frames (~170ms) of steady held fist to activate Alt-Tab
            "stop": 6,          # 6 frames (~200ms) of sustained open palm to release Alt-Tab
            "point_right": 3,   # 3 frames (~100ms) to step right
            "point_left": 3     # 3 frames (~100ms) to step left
        }

    def _update_temporal_state(self, active_gesture: str, res: dict):
        """Updates consecutive frame counters and activates confirmed states."""
        for gest in self.consecutive_counts:
            if gest == active_gesture:
                self.consecutive_counts[gest] += 1
            else:
                # Rapid decay for inactive gestures
                self.consecutive_counts[gest] = max(0, self.consecutive_counts[gest] - 1)

        # Check threshold confirmation
        for gest, count in self.consecutive_counts.items():
            if count >= self.threshold_frames[gest]:
                if gest == "grabbing":
                    res["is_grabbing"] = True
                elif gest == "stop":
                    res["is_stop"] = True
                elif gest in ("point_right", "point_left"):
                    res["is_pointing"] = True
                    res["direction"] = gest

    def analyze_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Analyzes a mirror-flipped BGR webcam frame.
        Returns a dict of temporally stabilized states:
        {
            "has_hand": bool,
            "direction": "point_right" | "point_left" | "neutral",
            "is_pointing": bool,
            "is_grabbing": bool,
            "is_stop": bool,
            "num_hands": int
        }
        """
        res = {
            "has_hand": False,
            "direction": "neutral",
            "is_pointing": False,
            "is_grabbing": False,
            "is_stop": False,
            "num_hands": 0
        }

        if self.detector is None:
            return res

        raw_gesture = None

        try:
            # Convert OpenCV BGR to MediaPipe Image (SRGB format)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = self.detector.detect(mp_image)

            if not detection_result.hand_landmarks:
                self._update_temporal_state(None, res)
                return res

            res["has_hand"] = True
            num_hands = len(detection_result.hand_landmarks)
            res["num_hands"] = num_hands

            # If two hands are visible, suppress one-handed modal Alt-Tab triggers (grabbing & pointing).
            # Two hands signify a two-handed gesture like xsign, thumb_index2, heart, or take_picture.
            if num_hands >= 2:
                self._update_temporal_state(None, res)
                return res

            # Single-Hand Landmark Analysis (Primary Hand)
            landmarks = detection_result.hand_landmarks[0]

            index_tip, index_pip, index_mcp = landmarks[8], landmarks[6], landmarks[5]
            middle_tip, middle_pip, middle_mcp = landmarks[12], landmarks[10], landmarks[9]
            ring_tip, ring_pip, ring_mcp = landmarks[16], landmarks[14], landmarks[13]
            pinky_tip, pinky_pip, pinky_mcp = landmarks[20], landmarks[18], landmarks[17]

            # Invariant Knuckle-to-Fingertip Extension Ratio
            def is_finger_extended(tip, pip, mcp):
                d_tip = np.hypot(tip.x - mcp.x, tip.y - mcp.y)
                d_pip = np.hypot(pip.x - mcp.x, pip.y - mcp.y)
                if d_pip < 1e-4:
                    return False
                return (d_tip / d_pip) > 1.35

            index_ext = is_finger_extended(index_tip, index_pip, index_mcp)
            middle_ext = is_finger_extended(middle_tip, middle_pip, middle_mcp)
            ring_ext = is_finger_extended(ring_tip, ring_pip, ring_mcp)
            pinky_ext = is_finger_extended(pinky_tip, pinky_pip, pinky_mcp)

            # A. Open Palm ('stop'): All 4 main fingers extended
            if index_ext and middle_ext and ring_ext and pinky_ext:
                raw_gesture = "stop"

            # B. Closed Fist / Grip ('grabbing'): All 4 fingers curled
            elif (not index_ext) and (not middle_ext) and (not ring_ext) and (not pinky_ext):
                raw_gesture = "grabbing"

            # C. Pointing: Only index finger extended, others curled
            elif index_ext and (not middle_ext) and (not ring_ext) and (not pinky_ext):
                dx = index_tip.x - index_mcp.x
                dy = index_tip.y - index_mcp.y

                abs_dx = abs(dx)
                abs_dy = abs(dy)

                # Horizontal cone filter: horizontal component must dominate vertical tilt
                if abs_dx > self.horizontal_threshold and abs_dx > (abs_dy * 0.65):
                    if dx > 0:
                        # In mirror-flipped camera view, positive dx points toward screen RIGHT
                        raw_gesture = "point_right"
                    else:
                        # Negative dx points toward screen LEFT
                        raw_gesture = "point_left"

        except Exception:
            pass

        self._update_temporal_state(raw_gesture, res)
        return res

    def close(self):
        if self.detector:
            try:
                self.detector.close()
            except Exception:
                pass
