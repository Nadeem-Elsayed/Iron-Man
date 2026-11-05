# STARK OS // IronMan v2.0.0

**Release Tag:** `v2.0.0`  
**Distribution Archive:** `dist/IronManClient-v2.0.0-Windows.zip` (111.8 MB)  
**Cloud Microservice:** AWS Lambda container in `us-east-2` (`version: 2.0.0`, 9 classes)

---

## Overview

IronMan v2.0.0 adds modal window management and browser tab dismissal, powered by edge MediaPipe landmarking on the CPU and an updated 9-class ResNet-50 classifier deployed to AWS Lambda.

---

## What's New

- **Modal Alt-Tab Navigation**:
  - Closed fist (`grabbing`): Launch and hold the Windows Task Switcher (`Alt + Tab`).
  - Index pointing (`point_right` / `point_left`): Cycle forward (`Tab`) or backward (`Shift + Tab`) through open windows.
  - Open palm (`stop`): Confirm window selection and release `Alt`.
  - Includes a 1.0s engagement lock and multi-frame smoothing to prevent premature cancellation.
- **Tab Purge (`xsign`)**:
  - Crossing forearms or hands sends `Ctrl + W` to close the active browser tab or document without window minimization conflicts.
- **9-Class ResNet-50 Classifier**:
  - Retrained on 10,794 balanced HaGRID frames with 98.11% accuracy.
  - Eliminates confusion between `xsign`, `mute`, and `thumb_index2`.
- **Stabilization & Transition Guarding**:
  - 0.90s global inter-action cooldown to eliminate flip-flopping during hand movements.
  - Two-handed guard suppresses single-handed Alt-Tab triggers when both hands enter the frame.
  - Multi-frame temporal consensus (2 to 5 frames) filters transient motions.
- **Standalone Client Fixes**:
  - Resolved MediaPipe PyInstaller runtime dependencies via an in-memory matplotlib hook.
  - Added one-click launcher `START_IRONMAN.bat`.

---

## Gesture Reference

| Gesture Class | Visual Pose | Desktop Action | Hotkey |
| :--- | :--- | :--- | :---: |
| `xsign` | Crossed forearms / wrists | Close active tab | `Ctrl + W` |
| `grabbing` | Closed fist | Hold Task Switcher | Hold `Alt + Tab` |
| `point` (Right) | Index finger right | Next window | `Tab` |
| `point` (Left) | Index finger left | Previous window | `Shift + Tab` |
| `stop` | Flat open palm | Select window / release | Release `Alt` / `Enter` |
| `thumb_index2` | L-shape hands | Show / minimize desktop | `Win + D` |
| `heart` | Two hands forming heart | Launch / focus Steam | `steam://open/main` |
| `mute` | Index finger over lips | Toggle master audio | `Volume Mute` |
| `take_photo` | Camera framing box | Windows screenshot | `Win + Shift + S` |
| `no_gesture` | Neutral hand | Passive observation | *None* |

---

## Quick Start

1. Download and extract `IronManClient-v2.0.0-Windows.zip`.
2. Run `START_IRONMAN.bat` (or `IronManClient.exe`).
3. Controls:
   - `[A]` Toggle actions on/off
   - `[P]` Toggle Picture-in-Picture pin
   - `[H]` Toggle hint overlay
   - `[T]` Cycle confidence threshold
   - `[Q]` Quit
   - `[Ctrl + Alt + Q]` Global emergency killswitch
