import os
import sys
import time
import random
import ctypes
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class IronManHUD:
    """
    Minimalist Stark Industries HUD:
    - Pure video overlays: Progress Bar (startup boot), IronInterface (ambient loop).
    - Dynamic action lock-on dials (IronManHud1) triggered on ANY gesture.
    - Zero UI clutter: all headers, online badges, FPS counters, debug menus, and popups removed.
    - Optional bottom-left shortcut hint toggleable via 'H' or --hide-hints.
    """

    def __init__(self, target_size=(480, 270), graphics_dir=None):
        self.target_w, self.target_h = target_size
        if graphics_dir is None:
            graphics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cool_graphics")
        self.graphics_dir = graphics_dir

        self.icon_png_path = os.path.join(self.graphics_dir, "IronManIcon.png")
        self.icon_ico_path = os.path.join(self.graphics_dir, "IronManIcon.ico")

        # Fonts (only used if show_hints is enabled)
        self._init_fonts()

        # Initialize and verify .ico file
        self._init_icon_files()

        # Pre-cache video frames in RAM
        self._init_video_caches()

        # State management
        self.boot_frame_idx = 0
        self.boot_complete = False
        self.interface_frame_idx = 0

        # Action dial lock-on effect state
        self.active_action_effect = None

    def _init_fonts(self):
        font_dir = r"C:\Windows\Fonts"
        try:
            self.font_hint = ImageFont.truetype(os.path.join(font_dir, "segoeui.ttf"), 10)
        except Exception:
            self.font_hint = ImageFont.load_default()

    def _init_icon_files(self):
        if os.path.exists(self.icon_png_path) and not os.path.exists(self.icon_ico_path):
            try:
                img = Image.open(self.icon_png_path)
                img.save(self.icon_ico_path, format="ICO", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
            except Exception as e:
                print(f"[HUD] Icon generation notice: {e}")

    def set_window_icon(self, hwnd: int):
        if not hwnd or not os.path.exists(self.icon_ico_path):
            return
        try:
            WM_SETICON = 0x0080
            IMAGE_ICON = 1
            LR_LOADFROMFILE = 0x0010

            hicon_big = ctypes.windll.user32.LoadImageW(
                None,
                os.path.abspath(self.icon_ico_path),
                IMAGE_ICON,
                32,
                32,
                LR_LOADFROMFILE
            )
            hicon_small = ctypes.windll.user32.LoadImageW(
                None,
                os.path.abspath(self.icon_ico_path),
                IMAGE_ICON,
                16,
                16,
                LR_LOADFROMFILE
            )

            if hicon_big:
                ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, 1, hicon_big)
                try:
                    SetClassLongPtr = getattr(ctypes.windll.user32, 'SetClassLongPtrW', ctypes.windll.user32.SetClassLongW)
                    SetClassLongPtr(hwnd, -14, hicon_big)   # GCLP_HICON = -14
                except Exception:
                    pass

            if hicon_small:
                ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, 0, hicon_small)
                try:
                    SetClassLongPtr = getattr(ctypes.windll.user32, 'SetClassLongPtrW', ctypes.windll.user32.SetClassLongW)
                    SetClassLongPtr(hwnd, -34, hicon_small) # GCLP_HICONSM = -34
                except Exception:
                    pass
        except Exception as e:
            print(f"[HUD] set_window_icon notice: {e}")

    def _init_video_caches(self):
        # 1. Main Ambient Interface Loop (IronInterface.webm)
        self.interface_frames = []
        if_path = os.path.join(self.graphics_dir, "IronInterface.webm")
        if os.path.exists(if_path):
            cap = cv2.VideoCapture(if_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self.interface_frames.append(cv2.resize(frame, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA))
            cap.release()

        # 2. Startup Boot Sequence (Progress Bar.webm)
        self.pb_frames = []
        pb_path = os.path.join(self.graphics_dir, "Progress Bar.webm")
        if os.path.exists(pb_path):
            cap = cv2.VideoCapture(pb_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self.pb_frames.append(cv2.resize(frame, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA))
            cap.release()

        # 3. Action HUD Lock-on Dial (IronManHud1.webm)
        # Bounding box of circular dial: y=[255:845], x=[678:1198]
        self.hud_dial_primary = []
        self.hud_dial_secondary = []
        hud_path = os.path.join(self.graphics_dir, "IronManHud1.webm")
        if os.path.exists(hud_path):
            cap = cv2.VideoCapture(hud_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                crop = frame[255:845, 678:1198]
                # Skip leading blank/black frames so animation is immediately visible
                if crop.mean() > 2.0:
                    self.hud_dial_primary.append(cv2.resize(crop, (80, 80), interpolation=cv2.INTER_AREA))
                    self.hud_dial_secondary.append(cv2.resize(crop, (50, 50), interpolation=cv2.INTER_AREA))
            cap.release()

    def trigger_action_hud(self, gesture_name: str):
        """
        Spawns dynamic HUD lock-on dials at randomized preset anchor coordinates.
        Activates for every type of gesture.
        """
        if not self.hud_dial_primary:
            return

        w, h = self.target_w, self.target_h
        primary_w, primary_h = 80, 80
        sec_w, sec_h = 50, 50

        # 4 Pre-defined anchor quadrants
        zones = [
            {"x": w - primary_w - 20, "y": h - primary_h - 25, "id": "BR"},
            {"x": 20, "y": 30, "id": "TL"},
            {"x": w - primary_w - 20, "y": 30, "id": "TR"},
            {"x": 20, "y": h - primary_h - 25, "id": "BL"},
        ]

        primary_idx = random.randint(0, len(zones) - 1)
        sub_candidates = [i for i in range(len(zones)) if i != primary_idx]
        sub_idx = random.choice(sub_candidates)

        zone_primary = dict(zones[primary_idx])
        zone_secondary = dict(zones[sub_idx])

        if zone_secondary["id"] == "BR":
            zone_secondary["x"] = w - sec_w - 20
            zone_secondary["y"] = h - sec_h - 25
        elif zone_secondary["id"] == "TR":
            zone_secondary["x"] = w - sec_w - 20
            zone_secondary["y"] = 30

        self.active_action_effect = {
            "gesture": gesture_name,
            "start_time": time.time(),
            "duration": 2.0,
            "zone_primary": zone_primary,
            "zone_secondary": zone_secondary
        }

    def _blend_overlay_at(self, canvas: np.ndarray, overlay_patch: np.ndarray, x: int, y: int, opacity: float = 1.0):
        ph, pw = overlay_patch.shape[:2]
        ch, cw = canvas.shape[:2]

        if x >= cw or y >= ch or x + pw <= 0 or y + ph <= 0:
            return

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(cw, x + pw)
        y2 = min(ch, y + ph)

        ox1 = x1 - x
        oy1 = y1 - y
        ox2 = ox1 + (x2 - x1)
        oy2 = oy1 + (y2 - y1)

        patch_roi = overlay_patch[oy1:oy2, ox1:ox2]
        canvas_roi = canvas[y1:y2, x1:x2]

        if opacity < 0.99:
            patch_scaled = (patch_roi.astype(np.float32) * opacity).astype(np.uint8)
            cv2.add(canvas_roi, patch_scaled, dst=canvas_roi)
        else:
            cv2.add(canvas_roi, patch_roi, dst=canvas_roi)

    def render(
        self,
        frame: np.ndarray,
        show_hints: bool = True
    ) -> np.ndarray:
        """
        Renders the video frame with only:
        1. Startup Progress Bar (boot sequence).
        2. Ambient IronInterface.webm full visor overlay.
        3. Dynamic IronManHud1.webm action dials on gesture.
        4. Optional bottom-left hint text (toggleable via 'H').
        Zero debug menus, zero top bars, zero FPS text, zero popups.
        """
        if frame.shape[1] != self.target_w or frame.shape[0] != self.target_h:
            canvas = cv2.resize(frame, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        else:
            canvas = frame.copy()

        # 1. Startup Boot Sequence (Progress Bar.webm)
        if not self.boot_complete and self.pb_frames:
            pb_frame = self.pb_frames[self.boot_frame_idx]
            canvas = cv2.add(canvas, pb_frame)
            self.boot_frame_idx += 1
            if self.boot_frame_idx >= len(self.pb_frames):
                self.boot_complete = True
            return canvas

        # 2. Ambient Continuous Iron Man Visor Loop (IronInterface.webm)
        if self.interface_frames:
            if_frame = self.interface_frames[self.interface_frame_idx]
            canvas = cv2.add(canvas, if_frame)
            self.interface_frame_idx = (self.interface_frame_idx + 1) % len(self.interface_frames)

        # 3. Dynamic Action Lock-On Dials (IronManHud1.webm)
        if self.active_action_effect is not None:
            eff = self.active_action_effect
            elapsed = time.time() - eff["start_time"]
            if elapsed > eff["duration"]:
                self.active_action_effect = None
            else:
                remaining = eff["duration"] - elapsed
                opacity = min(1.0, remaining / 0.4) if remaining < 0.4 else 1.0

                frame_idx = int(elapsed * 30.0) % len(self.hud_dial_primary)

                dial_prim = self.hud_dial_primary[frame_idx]
                zp = eff["zone_primary"]
                self._blend_overlay_at(canvas, dial_prim, zp["x"], zp["y"], opacity=opacity)

                dial_sec = self.hud_dial_secondary[frame_idx]
                zs = eff["zone_secondary"]
                self._blend_overlay_at(canvas, dial_sec, zs["x"], zs["y"], opacity=opacity * 0.85)

        # 4. Optional Subtle Bottom-Left Hints (if enabled)
        if show_hints:
            pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            hint_text = "[A] Actions   [P] Pin   [H] Hide   [Ctrl+Alt+Q] Exit"
            draw.text((14, self.target_h - 18), hint_text, font=self.font_hint, fill=(120, 135, 150))
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return canvas
