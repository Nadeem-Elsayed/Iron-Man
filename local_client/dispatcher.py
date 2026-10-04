import time
import webbrowser
import pyautogui

# Set safety margins to prevent desktop lockups
pyautogui.PAUSE = 0.05
# Slam the mouse to any corner of the screen to instantly abort all macros if things go wrong
pyautogui.FAILSAFE = True 

class ActionDispatcher:
    def __init__(self, cooldown: float = 1.2):
        """
        Initializes the dispatcher with a cooldown to debounce consecutive frame triggers.
        """
        self.cooldown = cooldown
        self.last_fired_time = 0.0

        # Action Mapping Dictionary
        # Keys correspond to the exact string labels output by your PyTorch inference model
        self.action_map = {
            # Window Management (Jester Dataset sweeping motions)
            "swipe_left": lambda: pyautogui.hotkey('alt', 'shift', 'tab'), # Previous Window
            "swipe_right": lambda: pyautogui.hotkey('alt', 'tab'),         # Next Window
            
            # Desktop & Utility (HaGRID Dataset static poses)
            "thumb_index2": lambda: pyautogui.hotkey('win', 'd'),          # Show/Hide Desktop
            "take_photo": lambda: pyautogui.hotkey('win', 'shift', 's'),   # OS Snipping Tool
            "mute": lambda: pyautogui.press('volumemute'),                 # Toggle Volume Mute
            
            # Browser Actions
            "heart": lambda: webbrowser.open_new_tab("https://www.instagram.com")
        }

    def execute(self, gesture_name: str) -> None:
        """
        Looks up the gesture in the action map and executes the mapped callable 
        if the cooldown period has elapsed.
        """
        now = time.time()
        
        # Suppress duplicate triggers if the cooldown hasn't cleared
        if (now - self.last_fired_time) < self.cooldown:
            return  

        # Fetch the mapped action and execute it
        action = self.action_map.get(gesture_name)
        if action:
            try:
                action()
                self.last_fired_time = now
                print(f"Executed: {gesture_name}")
            except Exception as e:
                print(f"Failed to execute {gesture_name}: {e}")
        else:
            # Catch-all for unrecognized classifications
            print(f"Unmapped gesture detected: {gesture_name}")

# Quick local test block
if __name__ == "__main__":
    print("Testing dispatcher locally. Simulating a 'heart' gesture in 3 seconds...")
    time.sleep(3)
    
    test_dispatcher = ActionDispatcher(cooldown=1.0)
    test_dispatcher.execute("heart")
    
    # Simulating a rapid duplicate trigger to test the debounce logic
    test_dispatcher.execute("heart")