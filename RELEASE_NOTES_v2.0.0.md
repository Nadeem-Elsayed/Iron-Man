# STARK OS // IronMan v2.0.0 — Modal Alt-Tab, Tab Purge & Hybrid Edge-Cloud Vision

> **Release Tag:** `v2.0.0`  
> **Release Target:** `main`  
> **Distribution Binary:** `dist/IronManClient-v2.0.0-Windows.zip` (111.8 MB)  
> **Target Cloud Microservice:** AWS Lambda container in `us-east-2` (`version: 2.0.0`, 9 classes)

---

## 🚀 Overview

**STARK OS // IronMan v2.0.0** is a major feature and performance release that expands gesture interactions from basic desktop shortcuts into a full-featured, interactive **modal window management system**. 

By pairing **MediaPipe Tasks HandLandmarker** on the edge CPU with a fine-tuned **9-class ResNet-50** deep classifier running in **AWS Lambda** (or directly on discrete GPUs via DirectML), v2.0.0 enables fluid task-switching, instant browser tab dismissal, and eliminates gesture collisions.

---

## ✨ What's New in v2.0.0

### 1. 🗂️ Interactive Modal Alt-Tab Task Switcher
* **Closed Fist Grip (`grabbing`)**: Make a fist to launch and lock the native Windows Task Switcher (`Alt + Tab`).
* **Directional Pointing Navigation (`point_left` / `point_right`)**: Point your index finger left or right to step forward or backward through your open windows with 1:1 precision.
* **Open Palm Confirmation (`stop`)**: Show a flat open palm to select the highlighted application and dismiss the switcher.
* **Engagement Lockout & Persistence**: A 1.0s engagement hold and multi-frame smoothing ensure your hand adjustments never prematurely dismiss your task switcher.

### 2. ❌ Instant Tab Purge (`xsign`)
* Form an **X** with your forearms or hands to instantly close the active browser tab, code editor tab, or document (`Ctrl + W`).
* Eliminates the v1.0 collision where crossing arms was misclassified as `thumb_index2` (Show Desktop).

### 3. 🧠 Re-Trained 9-Class ResNet-50 Deep Model
* Re-trained and validated across **10,794 balanced HaGRID dataset frames** with **98.11% accuracy**.
* **Zero cross-talk**: 100% recall on `xsign` with zero confusion into `mute` or `thumb_index2`.
* Synchronized and deployed to the AWS Lambda serverless microservice in `us-east-2`.

### 4. 🛡️ Transition Consensus & Two-Handed Guard
* **0.90s Global Inter-Action Cooldown**: Prevents unintended hotkey triggers during transition motions when lowering hands.
* **Two-Handed Guard**: Automatically suppresses single-handed Alt-Tab triggers (grabbing and pointing) whenever two hands are detected in the frame.
* **Temporal Consensus Filter**: Requires 2 to 5 consecutive frames before firing an action (`xsign`: 2 frames, `thumb_index2`: 3 frames, `mute`: 4 frames, `grabbing`: 5 frames).

### 5. ⚡ Standalone Client Packaging & Native Launcher
* Bundled standalone distribution: **`dist/IronManClient-v2.0.0-Windows.zip`** (111.8 MB).
* Resolved MediaPipe import dependencies via custom runtime hooks (`rthook_matplotlib.py`).
* Includes a one-click launcher: **`START_IRONMAN.bat`**.

---

## 📋 Complete Gesture & OS Action Reference

| Gesture Class | Pose Description | Desktop OS Action | Input Shortcut |
| :--- | :--- | :--- | :---: |
| **`xsign`** ⭐ *New* | Crossed forearms / wrists | Close active browser / editor tab | `Ctrl + W` |
| **`grabbing`** ⭐ *New* | Closed fist grip | Launch and hold Task Switcher | Hold `Alt + Tab` |
| **`point` (Right)** ⭐ *New* | Index finger pointing right | Next window in Task Switcher | `Tab` |
| **`point` (Left)** ⭐ *New* | Index finger pointing left | Previous window in Task Switcher | `Shift + Tab` |
| **`stop`** ⭐ *New* | Flat open palm | Confirm and select window | Release `Alt` / `Enter` |
| **`thumb_index2`** | L-shape / dual gun posture | Show / Minimize Desktop workspace | `Win + D` |
| **`heart`** | Two hands forming heart | Launch / Focus Steam client | `steam://open/main` |
| **`mute`** | Index finger over closed lips | Toggle master Windows audio | `Volume Mute / Unmute` |
| **`take_photo`** | Camera framing box | Windows Snipping Tool screenshot | `Win + Shift + S` |
| **`no_gesture`** | Relaxed / neutral hand | Passive observation mode | *None* |

> 📷 **Reference Images**: High-confidence visual references for each gesture are available in [`example_gestures/`](example_gestures/).

---

## 🛠️ Quick Start

### Option A: Standalone Windows App (No Python Required)
1. Download and extract **`IronManClient-v2.0.0-Windows.zip`**.
2. Double-click **`START_IRONMAN.bat`** (or `IronManClient.exe`).
3. The Always-on-Top Stark OS HUD will anchor to the top-right corner of your screen and connect to the live AWS cloud endpoint.

### Option B: Run from Source (Discrete GPU DirectML Acceleration)
```powershell
# Activate environment
.\.venv\Scripts\activate

# Install requirements
pip install -r local_client/requirements.txt

# Launch client with local DirectML GPU acceleration (<2ms inference)
python local_client/capture.py
```

### In-App Controls
* `[A]` — Toggle OS Actions on/off (Observation vs. Active mode)
* `[P]` — Pin / Unpin Always-on-Top Picture-in-Picture mode
* `[H]` — Toggle shortcut hint legend
* `[T]` — Cycle confidence threshold (`0.70` $\rightarrow$ `0.95`)
* `[Q]` — Quit application
* `[Ctrl + Alt + Q]` — **Global emergency killswitch** from any window

---

## 📦 Release Assets
* **`IronManClient-v2.0.0-Windows.zip`** — Standalone Windows 64-bit client executable with launcher.
* **`Source code (zip / tar.gz)`** — Full repository source including model training pipeline and cloud infrastructure.
