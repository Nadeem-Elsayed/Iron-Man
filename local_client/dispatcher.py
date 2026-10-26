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
    def __init__(self, cooldown: float = 1.2):
        """
        Initializes the dispatcher with a cooldown to debounce consecutive frame triggers.
        """
        self.cooldown = cooldown
        self.last_fired_time = 0.0
        self.desktop_minimized = False  # Track true bidirectional Show/Hide Desktop state

        # Action Mapping Dictionary
        # Modular, future-proof registry for gestures and their OS handlers
        self.action_map = {
            # Desktop & Utility
            "thumb_index2": self.toggle_desktop,                          # True Bidirectional Show/Hide Desktop
            "take_photo": take_fullscreen_screenshot,                      # Instant Fullscreen Screenshot
            "take_picture": take_fullscreen_screenshot,                    # Instant Fullscreen Screenshot alias
            "mute": lambda: pyautogui.press('volumemute'),                 # Toggle Volume Mute
            
            # Application Actions
            "heart": open_steam_and_switch,                                # Launch or Switch to Steam
            "hand_heart": open_steam_and_switch,                           # Launch or Switch to Steam alias

            # Window Management (Reserved for future gestures / Alt+Tab)
            "swipe_left": lambda: (pyautogui.hotkey('alt', 'shift', 'tab'), restore_ironman_hud()),
            "swipe_right": lambda: (pyautogui.hotkey('alt', 'tab'), restore_ironman_hud())
        }

    def toggle_desktop(self):
        """
        True instantaneous bidirectional Desktop Toggle (Win+D) keeping HUD pinned.
        """
        user32 = ctypes.windll.user32
        # Instant native Win + D keystroke (0 ms)
        user32.keybd_event(0x5B, 0, 0, 0)
        user32.keybd_event(0x44, 0, 0, 0)
        user32.keybd_event(0x44, 0, 2, 0)
        user32.keybd_event(0x5B, 0, 2, 0)

        self.desktop_minimized = not self.desktop_minimized
        time.sleep(0.04)
        restore_ironman_hud()
        print(f"[IronMan] Desktop Toggled ({'Minimized' if self.desktop_minimized else 'Restored'})")

    def execute(self, gesture_name: str) -> bool:
        """
        Looks up the gesture in the action map and executes the mapped callable 
        if the cooldown period has elapsed. Returns True if action fired, False otherwise.
        """
        now = time.time()
        
        # Suppress duplicate triggers if the cooldown hasn't cleared
        if (now - self.last_fired_time) < self.cooldown:
            return False

        # Fetch the mapped action and execute it
        action = self.action_map.get(gesture_name)
        if action:
            try:
                action()
                self.last_fired_time = now
                print(f"Executed: {gesture_name}")
                return True
            except Exception as e:
                print(f"Failed to execute {gesture_name}: {e}")
                return False
        else:
            # Catch-all for unrecognized classifications
            print(f"Unmapped gesture detected: {gesture_name}")
            return False


if __name__ == "__main__":
    print("Testing dispatcher locally. Simulating a 'thumb_index2' gesture in 2 seconds...")
    time.sleep(2)
    test_dispatcher = ActionDispatcher(cooldown=1.0)
    test_dispatcher.execute("thumb_index2")