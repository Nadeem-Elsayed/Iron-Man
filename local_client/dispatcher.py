import os
import time
import webbrowser
import ctypes
import subprocess
try:
    import winreg
except ImportError:
    winreg = None
import pyautogui

# Set safety margins to prevent desktop lockups
pyautogui.PAUSE = 0.05
# Slam the mouse to any corner of the screen to instantly abort all macros if things go wrong
pyautogui.FAILSAFE = True 


def restore_ironman_hud():
    """
    Guarantees the IronMan HUD is never buried, hidden, or minimized by OS actions.
    Restores the window and re-asserts HWND_TOPMOST so it stays anchored in the top-right.
    """
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, "STARK OS // IronMan Gesture HUD")
        if not hwnd:
            hwnd = user32.FindWindowW(None, "IronMan Gesture Client")
        if hwnd:
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            # HWND_TOPMOST = -1, SWP_NOSIZE = 1, SWP_NOMOVE = 2, SWP_NOACTIVATE = 16, SWP_SHOWWINDOW = 64 (0x0053)
            user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0053)
    except Exception:
        pass


def toggle_desktop_keep_hud():
    """
    Toggles Show/Hide Desktop (Win+D) while keeping the IronMan HUD pinned in view.
    """
    pyautogui.hotkey('win', 'd')
    time.sleep(0.08)
    restore_ironman_hud()


def take_fullscreen_screenshot():
    """
    Takes an instant full-screen screenshot, saves it directly to Pictures/Screenshots,
    and flashes native Win+PrtScn for instant OS visual feedback without manual snip prompts.
    """
    try:
        screenshots_dir = os.path.join(os.path.expanduser("~"), "Pictures", "Screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(screenshots_dir, f"Screenshot_{timestamp}.png")
        pyautogui.screenshot(filepath)
        pyautogui.hotkey('win', 'printscreen')
        print(f"[IronMan] Fullscreen screenshot saved to: {filepath}")
    except Exception as e:
        print(f"[IronMan] Screenshot capture failed: {e}")
    finally:
        restore_ironman_hud()


def open_url_and_switch(url: str, app_keywords=("instagram", "chrome", "edge", "firefox", "brave", "opera")):
    """
    Opens URL in default browser AND brings the browser window directly to the foreground,
    while keeping the IronMan HUD pinned on top.
    """
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    # 1. Allow the target browser to take the foreground
    try:
        user32.AllowSetForegroundWindow(-1)
    except Exception:
        pass

    # 2. Open the URL
    webbrowser.open_new_tab(url)
    time.sleep(0.15)

    # 3. Locate and switch to the browser window
    try:
        matched_hwnds = []
        def enum_cb(hwnd, _):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buff, length + 1)
                    title = buff.value.lower()
                    for kw in app_keywords:
                        if kw in title and "ironman" not in title:
                            matched_hwnds.append(hwnd)
                            return 0
            return 1

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
        user32.EnumWindows(WNDENUMPROC(enum_cb), None)

        if matched_hwnds:
            target = matched_hwnds[0]
            if user32.IsIconic(target):
                user32.ShowWindow(target, 9)  # SW_RESTORE
            else:
                user32.ShowWindow(target, 5)  # SW_SHOW

            # Attach thread input to bypass Windows focus lock
            fg_hwnd = user32.GetForegroundWindow()
            fg_thread = user32.GetWindowThreadProcessId(fg_hwnd, None)
            cur_thread = kernel32.GetCurrentThreadId()

            if fg_thread != cur_thread and fg_thread != 0:
                user32.AttachThreadInput(cur_thread, fg_thread, True)
                user32.SetForegroundWindow(target)
                user32.BringWindowToTop(target)
                user32.AttachThreadInput(cur_thread, fg_thread, False)
            else:
                user32.SetForegroundWindow(target)
                user32.BringWindowToTop(target)

            print(f"[IronMan] Focused browser window (HWND: {target})")
    except Exception as e:
        print(f"[IronMan Window Manager] Focus switch notice: {e}")

    # Re-assert IronMan HUD stays on top above the new foreground browser
    time.sleep(0.08)
    restore_ironman_hud()


def open_steam_and_switch():
    """
    Opens or switches to the Steam application, brings it directly to the foreground,
    while keeping the IronMan HUD pinned on top.
    """
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    # 1. Allow target window to take the foreground
    try:
        user32.AllowSetForegroundWindow(-1)
    except Exception:
        pass

    # 2. Launch or unminimize Steam via steam:// protocol or direct executable
    opened = False
    try:
        os.startfile("steam://open/main")
        opened = True
    except Exception:
        pass

    if not opened:
        try:
            webbrowser.open("steam://open/main")
            opened = True
        except Exception:
            pass

    if not opened:
        # Fallback: Locate steam.exe via Windows Registry or standard paths
        steam_exe = None
        if winreg is not None:
            for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                for subkey in (r"Software\Valve\Steam", r"Software\WOW6432Node\Valve\Steam"):
                    try:
                        with winreg.OpenKey(root, subkey) as key:
                            val, _ = winreg.QueryValueEx(key, "SteamExe")
                            if val and os.path.exists(val):
                                steam_exe = val
                                break
                    except OSError:
                        continue
                if steam_exe:
                    break

        if not steam_exe:
            candidates = [
                r"C:\Program Files (x86)\Steam\steam.exe",
                r"C:\Program Files\Steam\steam.exe",
                os.path.expandvars(r"%ProgramFiles(x86)%\Steam\steam.exe"),
                os.path.expandvars(r"%ProgramFiles%\Steam\steam.exe")
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    steam_exe = candidate
                    break

        if steam_exe:
            try:
                subprocess.Popen([steam_exe, "steam://open/main"])
                opened = True
            except Exception as e:
                print(f"[IronMan] Steam executable launch error: {e}")

    time.sleep(0.15)

    # 3. Locate and switch to the Steam window
    try:
        matched_hwnds = []

        def enum_cb(hwnd, _):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buff, length + 1)
                    title = buff.value.lower()
                    if "ironman" not in title:
                        if title == "steam" or title.startswith("steam"):
                            matched_hwnds.append(hwnd)
                            return 0
                        # Check process name if title is not explicitly 'Steam'
                        pid = ctypes.c_ulong()
                        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                        if pid.value:
                            h_proc = kernel32.OpenProcess(0x1000, False, pid.value)
                            if h_proc:
                                try:
                                    pbuf = ctypes.create_unicode_buffer(1024)
                                    psize = ctypes.c_ulong(1024)
                                    if kernel32.QueryFullProcessImageNameW(h_proc, 0, pbuf, ctypes.byref(psize)):
                                        pname = os.path.basename(pbuf.value).lower()
                                        if pname in ("steam.exe", "steamwebhelper.exe"):
                                            matched_hwnds.append(hwnd)
                                            return 0
                                finally:
                                    kernel32.CloseHandle(h_proc)
            return 1

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
        user32.EnumWindows(WNDENUMPROC(enum_cb), None)

        if matched_hwnds:
            target = matched_hwnds[0]
            if user32.IsIconic(target):
                user32.ShowWindow(target, 9)  # SW_RESTORE
            else:
                user32.ShowWindow(target, 5)  # SW_SHOW

            # Attach thread input to bypass Windows focus lock
            fg_hwnd = user32.GetForegroundWindow()
            fg_thread = user32.GetWindowThreadProcessId(fg_hwnd, None)
            cur_thread = kernel32.GetCurrentThreadId()

            if fg_thread != cur_thread and fg_thread != 0:
                user32.AttachThreadInput(cur_thread, fg_thread, True)
                user32.SetForegroundWindow(target)
                user32.BringWindowToTop(target)
                user32.AttachThreadInput(cur_thread, fg_thread, False)
            else:
                user32.SetForegroundWindow(target)
                user32.BringWindowToTop(target)

            print(f"[IronMan] Focused Steam window (HWND: {target})")
    except Exception as e:
        print(f"[IronMan Window Manager] Steam focus switch notice: {e}")

    # Re-assert IronMan HUD stays on top above the new foreground Steam window
    time.sleep(0.08)
    restore_ironman_hud()


class ActionDispatcher:
    def __init__(self, cooldown: float = 1.0):
        """
        Initializes the dispatcher with per-gesture cooldowns and modal Alt-Tab state tracking.
        """
        self.default_cooldown = cooldown
        self.last_fired_times = {}       # gesture_name -> last trigger timestamp
        self.desktop_minimized = False   # Track true bidirectional Show/Hide Desktop state

        # Modal Alt-Tab State Machine
        self.alt_tab_active = False
        self.alt_tab_last_activity = 0.0
        self.alt_tab_start_time = 0.0
        self.alt_tab_timeout = 7.0       # Auto-release Alt after 7s of inactivity
        self.alt_tab_lockout = 1.0       # Minimum duration (1.0s) before 'stop' is allowed to release Alt-Tab

        # Global Inter-Action Cooldown (prevents transition flip-flopping across different gestures)
        self.last_global_action_time = 0.0
        self.global_cooldown = 0.90      # 900ms guard between ANY two distinct OS actions

        # Per-gesture cooldowns (in seconds)
        self.gesture_cooldowns = {
            "point_left": 0.55,          # Deliberate window step (prevents overshooting)
            "point_right": 0.55,         # Deliberate window step (prevents overshooting)
            "swipe_left": 0.50,          # Deliberate dynamic swipe step
            "swipe_right": 0.50,         # Deliberate dynamic swipe step
            "grabbing": 0.8,             # Trigger modal Alt-Tab
            "grip": 0.8,                 # Alias trigger
            "stop": 0.5,                 # Release modal Alt-Tab
            "palm": 0.5,                 # Alias release
            "xsign": 1.0,                # Close Tab (Ctrl + W)
            "thumb_index2": 1.2,         # Show/Hide Desktop
            "take_photo": 1.5,           # Fullscreen screenshot
            "take_picture": 1.5,         # Screenshot alias
            "mute": 0.8,                 # Volume mute toggle
            "heart": 2.0,                # Steam launcher
            "hand_heart": 2.0            # Steam launcher alias
        }

        # Action Mapping Dictionary
        self.action_map = {
            # Desktop & Utility
            "thumb_index2": self.toggle_desktop,
            "take_photo": take_fullscreen_screenshot,
            "take_picture": take_fullscreen_screenshot,
            "mute": lambda: pyautogui.press('volumemute'),

            # Tab & Window Closing
            "xsign": self.close_tab,

            # Modal Alt-Tab Task Switcher
            "grabbing": self.start_alt_tab,
            "grip": self.start_alt_tab,
            "point_right": self.step_alt_tab_right,
            "point_left": self.step_alt_tab_left,
            "stop": self.end_alt_tab,
            "palm": self.end_alt_tab,

            # Dynamic Swipes (also wired to Alt-Tab stepping)
            "swipe_left": self.step_alt_tab_left,
            "swipe_right": self.step_alt_tab_right,

            # Application Launchers
            "heart": open_steam_and_switch,
            "hand_heart": open_steam_and_switch
        }

    def _check_alt_tab_timeout(self):
        """Auto-releases Alt if user stopped interacting or walked away."""
        if self.alt_tab_active and (time.time() - self.alt_tab_last_activity > self.alt_tab_timeout):
            print("[IronMan Action] Alt-Tab timeout exceeded (7s). Auto-releasing Alt modifier.")
            self.end_alt_tab()

    def start_alt_tab(self):
        """
        Initiates modal Alt-Tab window switcher via native Win32 events.
        Holds Alt down and presses Tab once, keeping the switcher overlay visible.
        Enforces a 1.0s lockout preventing premature stop release.
        """
        if self.alt_tab_active:
            return

        user32 = ctypes.windll.user32
        now = time.time()
        try:
            # Hold Alt (0x12)
            user32.keybd_event(0x12, 0, 0, 0)
            time.sleep(0.04)
            # Tap Tab (0x09)
            user32.keybd_event(0x09, 0, 0, 0)
            user32.keybd_event(0x09, 0, 2, 0)
            self.alt_tab_active = True
            self.alt_tab_start_time = now
            self.alt_tab_last_activity = now
            print(f"[IronMan Action] Modal Alt-Tab ENGAGED (Holding Alt -- 'Stop' locked for {self.alt_tab_lockout:.1f}s)")
        except Exception as e:
            print(f"[IronMan Action] Alt-Tab start failed: {e}")
        finally:
            restore_ironman_hud()

    def step_alt_tab_right(self):
        """Steps forward in the window switcher (Tab). Only valid while Alt-Tab is active."""
        if not self.alt_tab_active:
            return

        now = time.time()
        self._check_alt_tab_timeout()
        user32 = ctypes.windll.user32
        try:
            user32.keybd_event(0x09, 0, 0, 0)
            user32.keybd_event(0x09, 0, 2, 0)
            self.alt_tab_last_activity = now
            print("[IronMan Action] Alt-Tab Step Forward >>")
        except Exception as e:
            print(f"[IronMan Action] Alt-Tab step forward failed: {e}")
        finally:
            restore_ironman_hud()

    def step_alt_tab_left(self):
        """Steps backward in the window switcher (Shift + Tab). Only valid while Alt-Tab is active."""
        if not self.alt_tab_active:
            return

        now = time.time()
        self._check_alt_tab_timeout()
        user32 = ctypes.windll.user32
        try:
            # Shift down (0x10)
            user32.keybd_event(0x10, 0, 0, 0)
            # Tab tap (0x09)
            user32.keybd_event(0x09, 0, 0, 0)
            user32.keybd_event(0x09, 0, 2, 0)
            # Shift up (0x10)
            user32.keybd_event(0x10, 0, 2, 0)
            self.alt_tab_last_activity = now
            print("[IronMan Action] << Alt-Tab Step Backward")
        except Exception as e:
            print(f"[IronMan Action] Alt-Tab step backward failed: {e}")
        finally:
            restore_ironman_hud()

    def end_alt_tab(self):
        """
        Ends modal Alt-Tab window switcher.
        Releases Alt key to focus and lock in the selected window.
        """
        if not self.alt_tab_active:
            return

        user32 = ctypes.windll.user32
        try:
            # Release Alt (0x12)
            user32.keybd_event(0x12, 0, 2, 0)
            self.alt_tab_active = False
            print("[IronMan Action] Modal Alt-Tab RELEASED (Window Focused)")
        except Exception as e:
            print(f"[IronMan Action] Alt-Tab release failed: {e}")
        finally:
            time.sleep(0.05)
            restore_ironman_hud()

    def close_tab(self):
        """
        Closes active browser tab / document window (Ctrl + W).
        Ensures any active Alt modifier is released first.
        If the HUD window is focused, targets the window directly beneath it.
        """
        if self.alt_tab_active:
            self.end_alt_tab()
        user32 = ctypes.windll.user32
        try:
            # If the foreground window is IronMan HUD, refocus the top active application underneath
            fg = user32.GetForegroundWindow()
            hud_hwnd = user32.FindWindowW(None, "STARK OS // IronMan Gesture HUD")
            if not hud_hwnd:
                hud_hwnd = user32.FindWindowW(None, "IronMan Gesture Client")
            if fg and hud_hwnd and fg == hud_hwnd:
                GW_HWNDNEXT = 2
                nxt = user32.GetWindow(hud_hwnd, GW_HWNDNEXT)
                while nxt:
                    if user32.IsWindowVisible(nxt) and not user32.IsIconic(nxt):
                        length = user32.GetWindowTextLengthW(nxt)
                        if length > 0:
                            user32.SetForegroundWindow(nxt)
                            time.sleep(0.04)
                            break
                    nxt = user32.GetWindow(nxt, GW_HWNDNEXT)

            # Ctrl down (0x11)
            user32.keybd_event(0x11, 0, 0, 0)
            # W tap (0x57)
            user32.keybd_event(0x57, 0, 0, 0)
            user32.keybd_event(0x57, 0, 2, 0)
            # Ctrl up (0x11)
            user32.keybd_event(0x11, 0, 2, 0)
            print("[IronMan Action] >>> CLOSE TAB EXECUTED: Sent Ctrl + W to active window <<<")
        except Exception as e:
            print(f"[IronMan Action] Close tab failed: {e}")
        finally:
            restore_ironman_hud()

    def toggle_desktop(self):
        """
        True instantaneous bidirectional Desktop Toggle (Win+D) keeping HUD pinned.
        """
        if self.alt_tab_active:
            self.end_alt_tab()

        # Suppress accidental desktop minimizing if xsign was recently fired
        if (time.time() - self.last_fired_times.get("xsign", 0.0)) < 1.0:
            print("[IronMan Action] Suppressed false thumb_index2 (xsign protection active)")
            return False

        user32 = ctypes.windll.user32
        # Instant native Win + D keystroke (0 ms)
        user32.keybd_event(0x5B, 0, 0, 0)
        user32.keybd_event(0x44, 0, 0, 0)
        user32.keybd_event(0x44, 0, 2, 0)
        user32.keybd_event(0x5B, 0, 2, 0)

        self.desktop_minimized = not self.desktop_minimized
        time.sleep(0.04)
        restore_ironman_hud()
        print(f"[IronMan Action] Desktop Toggled ({'Minimized' if self.desktop_minimized else 'Restored'})")
        return True

    def execute(self, gesture_name: str) -> bool:
        """
        Looks up the gesture in the action map and executes the mapped handler
        with strict modal state gating, engagement lockout, and per-gesture cooldowns.
        """
        now = time.time()
        self._check_alt_tab_timeout()

        # Strict State Gating:
        if self.alt_tab_active:
            # Protect against premature stop when unclenching fist
            if gesture_name in ("stop", "palm"):
                elapsed = now - getattr(self, "alt_tab_start_time", 0.0)
                if elapsed < self.alt_tab_lockout:
                    rem = self.alt_tab_lockout - elapsed
                    if (now - self.last_fired_times.get("suppressed_stop_notice", 0.0)) > 0.4:
                        self.last_fired_times["suppressed_stop_notice"] = now
                        print(f"[IronMan Action] Alt-Tab lockout active ({rem:.2f}s remaining before 'Stop' allowed)")
                    return False

            # While in Alt-Tab mode, ONLY allow navigation and release
            if gesture_name not in ("point_left", "point_right", "stop", "palm", "swipe_left", "swipe_right"):
                return False
        else:
            # While NOT in Alt-Tab mode, disallow navigation and release
            if gesture_name in ("point_left", "point_right", "stop", "palm"):
                return False

        # Determine cooldown for this specific gesture
        cd = self.gesture_cooldowns.get(gesture_name, self.default_cooldown)
        last_time = self.last_fired_times.get(gesture_name, 0.0)

        # Suppress duplicate triggers if per-gesture cooldown has not elapsed
        if (now - last_time) < cd:
            return False

        # Global inter-action cooldown (suppresses transition flip-flopping across different gestures)
        # Note: modal Alt-Tab stepping (point_left/point_right) is exempt to keep navigation fluid
        is_alt_tab_step = self.alt_tab_active and (gesture_name in ("point_left", "point_right", "swipe_left", "swipe_right"))
        if not is_alt_tab_step:
            if (now - self.last_global_action_time) < self.global_cooldown:
                return False

        action = self.action_map.get(gesture_name)
        if action:
            try:
                ret = action()
                if ret is False:
                    return False
                self.last_fired_times[gesture_name] = now
                if not is_alt_tab_step:
                    self.last_global_action_time = now
                print(f"[IronMan Action] Executed: {gesture_name}")
                return True
            except Exception as e:
                print(f"[IronMan Action] Failed to execute {gesture_name}: {e}")
                return False
        else:
            return False

    def cleanup(self):
        """Guarantees modifier keys are released upon exit."""
        if self.alt_tab_active:
            try:
                pyautogui.keyUp('alt')
            except Exception:
                pass
            self.alt_tab_active = False


if __name__ == "__main__":
    print("Testing dispatcher locally...")
    test_dispatcher = ActionDispatcher()
    print("Simulating grabbing (Start Alt-Tab)...")
    test_dispatcher.execute("grabbing")
    time.sleep(0.5)
    print("Simulating point_right (Step forward)...")
    test_dispatcher.execute("point_right")
    time.sleep(0.5)
    print("Simulating stop (Release Alt-Tab)...")
    test_dispatcher.execute("stop")
    test_dispatcher.cleanup()