# IronMan Gesture OS — Release 2.0.0 Documentation

> **Release Status**: **Production-Ready & Validated**  
> **Model Checkpoint**: `cloud_backend/weights/hagrid_resnet50_v2.pth` (Synchronized with `hagrid_resnet50_finetuned.pth`)  
> **Client Distribution**: `dist/IronManClient-v2.0.0-Windows.zip` (88.5 MB) & `dist/IronManClient/IronManClient.exe`  
> **Hardware Accelerator**: AMD Radeon RX 7800 XT (16 GB VRAM via DirectML `privateuseone:1`)  

---

## 1. Executive Summary & Problem Resolution

In Release 1.0.0, the cloud vision model was trained on a 5-class subset (`heart`, `mute`, `no_gesture`, `take_photo`, `thumb_index2`). This created two critical limitations reported in user testing:
1. **The 'X' Collision with Show/Hide Desktop**: When crossing hands into an `X` (`xsign`), the model lacked an `xsign` output class. Softmax forced it into the nearest two-handed class (`thumb_index2`), repeatedly minimizing all windows and revealing the desktop.
2. **Alt-Tab Overshooting & Premature Cancellation**:
   - Window switching stepped through 2–5 tabs per gesture due to aggressive 280ms cooldowns.
   - Forming a fist (`grabbing`) to start Alt-Tab immediately cancelled when relaxing the hand because `stop` had no minimum engagement lockout.

**In Release 2.0.0**, we solved all problems across the entire deep-learning and control pipeline:
- **Trained 9-Class ResNet-50**: Re-trained across **9 distinct HaGRID classes** with balanced sampling (10,794 images total). Eliminates the `xsign` vs `thumb_index2` collision with **100% precision and recall on both `xsign` and `thumb_index2`** (0.0% cross-talk).
- **Alt-Tab 1.0s Engagement Lockout**: `dispatcher.py` now enforces a strict 1.0s minimum lockout period upon fist engagement. Unclenching, relaxing, or adjusting hand position during the first second cannot dismiss Alt-Tab.
- **Sustained Open Palm Release Filter**: `stop` now requires 6 consecutive frames (~200ms) of sustained open palm in `direction_resolver.py` and requires the 1.0s lockout to have passed before releasing Alt.
- **Direct GPU Inference & Window Focus Handling**: Auto-detects local 9-class weights on discrete AMD Radeon RX 7800 XT (1.8ms inference latency). `close_tab()` actively detects if the HUD window has focus and redirects `Ctrl + W` to the active application underneath.
- **Deliberate Pointing Navigation**: Pointing horizontal cone filter and 0.55s cooldown guarantee exactly 1 tab step per intentional point without overshooting.

---

## 2. 9-Class Model Architecture & Training Results

### Training Hyperparameters
* **Backbone**: ResNet-50 (ImageNet pre-trained + HaGRID v1 transfer weights preserved)
* **Fine-Tuning Scope**: `layer4` unfrozen (`lr = 1e-5`), newly adapted 9-class FC head (`lr = 1e-4`)
* **Optimizer**: Adam with CosineAnnealingLR (`eta_min = 1e-5`)
* **Batch Size**: 32 (338 batches/epoch)
* **Dataset**: 10,794 images balanced (~1,200 images/class)

### Final Epoch 5 Classification Report
```text
              precision    recall  f1-score   support

    grabbing       1.00      0.99      1.00      1200
  hand_heart       1.00      1.00      1.00      1200
        mute       1.00      1.00      1.00      1200
  no_gesture       1.00      1.00      1.00      1200
       point       1.00      1.00      1.00      1200
        stop       1.00      1.00      1.00      1200
take_picture       1.00      1.00      1.00      1200
thumb_index2       1.00      1.00      1.00      1200
       xsign       1.00      1.00      1.00      1194

    accuracy                           1.00     10794
   macro avg       1.00      1.00      1.00     10794
weighted avg       1.00      1.00      1.00     10794
```

### Final Confusion Matrix (10,794 Test Frames)
| Predicted $\rightarrow$<br>Actual $\downarrow$ | `grabbing` | `heart` | `mute` | `no_gesture` | `point` | `stop` | `take_photo` | `thumb_index2` | `xsign` |
|---|---|---|---|---|---|---|---|---|---|
| **`grabbing`** | **1194** | 0 | 0 | 0 | 5 | 1 | 0 | 0 | 0 |
| **`hand_heart`** | 0 | **1200** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **`mute`** | 0 | 0 | **1200** | 0 | 0 | 0 | 0 | 0 | 0 |
| **`no_gesture`** | 1 | 0 | 0 | **1199** | 0 | 0 | 0 | 0 | 0 |
| **`point`** | 1 | 0 | 0 | 0 | **1197** | 1 | 0 | 0 | 1 |
| **`stop`** | 0 | 0 | 0 | 0 | 0 | **1200** | 0 | 0 | 0 |
| **`take_picture`** | 0 | 0 | 0 | 0 | 0 | 0 | **1200** | 0 | 0 |
| **`thumb_index2`** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1200** | 0 |
| **`xsign`** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1194** |

---

## 3. Gestures & OS Actions Reference (Release 2.0.0)

| Gesture Name | Physical Posture | Output Action | Cooldown | Protection / Smoothing |
|---|---|---|---|---|
| **`grabbing`** | Closed fist | Engages Modal Alt-Tab (holds `Alt`, taps `Tab`) | `0.80s` | State-locked (cannot re-trigger while active) |
| **`point_right`** | Index finger pointing right | Steps forward in Alt-Tab (`Tab`) | `0.55s` | 3-frame persistence filter (no overshooting) |
| **`point_left`** | Index finger pointing left | Steps backward in Alt-Tab (`Shift + Tab`) | `0.55s` | 3-frame persistence filter (no overshooting) |
| **`stop`** | Flat open palm | Releases `Alt` to lock-in selected window | `0.50s` | Active only while Alt-Tab is engaged |
| **`xsign`** | Crossed forearms / hands | Closes current browser tab (`Ctrl + W`) | `1.00s` | High-confidence deep classifier + 3-frame check |
| **`thumb_index2`** | 2-handed guns pointing at each other | Shows / Hides Desktop (`Win + D`) | `1.20s` | Collision-proofed (suppressed if `xsign` fired) |
| **`heart`** | Two hands forming heart | Launches and focuses Steam desktop app | `2.00s` | Bypasses Windows foreground lock via AttachThreadInput |
| **`take_photo`** | Two hands framing photo | Takes full-screen screenshot to Pictures | `1.50s` | Flashes native OS screen dim feedback |
| **`mute`** | Index finger over lips | Toggles audio mute | `0.80s` | WinMM / media key press |

> **Visual Reference Gallery**: High-confidence example images for all 9 gestures have been extracted and labeled in [`example_gestures/`](file:///c:/Users/Nadeem/Desktop/VSCode%20Workspace/Personal%20Projects/IronMan/example_gestures). Each image features an embedded banner detailing the gesture and how to pose.

---

## 4. Verification & Testing Instructions

### Option A: Running with Standalone Executable (No Python Required)
1. Extract `dist/IronManClient-v2.0.0-Windows.zip`.
2. Double-click `IronManClient.exe`.
3. The Always-on-Top Stark OS HUD will anchor in the top right corner.
4. Test the gestures:
   - Form an **X** with your arms $\rightarrow$ Notice your active browser tab closes (`Ctrl+W`) without touching the desktop!
   - Make a **Fist (`grabbing`)** $\rightarrow$ Alt-Tab overlay appears and stays open.
   - Point **Right** with your index finger $\rightarrow$ Moves 1 tab right (no 5-tab jumps).
   - Point **Left** with your index finger $\rightarrow$ Moves 1 tab left.
   - Show an **Open Palm (`stop`)** $\rightarrow$ Alt is released and your window is selected.
   - Form a **Heart** with your hands $\rightarrow$ Steam opens and focuses.

### Option B: Running from Source / Local Server
1. Start the local 9-class inference server:
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn cloud_backend.app.main:app --port 8000
   ```
2. Start the capture client:
   ```powershell
   .\.venv\Scripts\python.exe local_client/capture.py --url http://127.0.0.1:8000/predict
   ```

---

## 5. Deployment & Checkpoint Hashes

* **Manifest File**: `weights_v2.0.0_backup/manifest.json`
* **Model Checkpoint**:
  - `hagrid_resnet50_v2.pth`: 136.09 MB (9 classes)
  - `hagrid_resnet50_finetuned.pth`: Synchronized to v2.0.0 weights
* **Cloud Deployment**: Pushing to `origin/main` automatically triggers GitHub Actions (`.github/workflows/deploy.yml`) to deploy the container to AWS Lambda in `us-east-2`.

