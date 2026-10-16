# IronMan Project: Context & Architecture Handoff

## 1. Project Overview & End Goal
**IronMan** is a real-time, dual-stream hand gesture recognition system that translates webcam feeds into OS-level desktop shortcuts and browser actions (via `pyautogui` and `webbrowser`), designed with a decoupled local-client / cloud-backend architecture.

### Target Gesture Streams & Action Mapping
1. **2D Static Stream (HaGRID Dataset - 5 Classes):**
   - `thumb_index2`: Show/Hide Desktop (`win + d`)
   - `take_photo` / `take_picture`: OS Snipping Tool (`win + shift + s`)
   - `mute`: Toggle Volume Mute (`volumemute`)
   - `heart` / `hand_heart`: Open new browser tab (Instagram)
   - `no_gesture`: Control/Resting state (No action)

2. **3D Dynamic Stream (20BN-Jester Dataset - 4 Classes):**
   - `0` -> `Swiping Left` (`swipe_left`): Previous Window (`alt + shift + tab`)
   - `1` -> `Swiping Right` (`swipe_right`): Next Window (`alt + tab`)
   - `2` -> `No gesture`: Static Control/Resting state (No action)
   - `3` -> `Doing other things`: Dynamic Control/Non-gesture movement state (No action)
   *(Note: "Swiping Down" was previously tested as a 3rd class but was removed because it is not mapped in the dispatcher and caused massive class-imbalance false positives).*

---

## 2. CRITICAL Hardware & Environment Constraints (DO NOT VIOLATE)
- **Local Hardware:** AMD Radeon RX 7800 XT (`gfx1101`) + AMD Ryzen 5 7600 (has iGPU) on Windows 11.
- **Why Not WSL2 / ROCm:** AMD Windows drivers lock out `/dev/kfd` passthrough in WSL2 for `gfx1101`. Do NOT attempt to switch hardware routing to CUDA or ROCm for local training.
- **Active Execution Path:** Native Windows Python 3.10 virtual environment (`.venv`) using **DirectML**:
  - `torch==2.4.1`, `torchvision==0.19.1`, `torch-directml==0.2.5.dev240914`
- **Device Selection:** DirectML Device `0` is the Ryzen integrated graphics. Device `1` is the discrete **RX 7800 XT**. Hardware routing in `train.py` must explicitly scan `torch_directml.device_name(i)` and lock onto the 7800 XT (`privateuseone:1`).
- **DataLoader Safety:** `pin_memory=False` and `num_workers=4` with `persistent_workers=True` in `dataset.py` are strictly required to prevent Windows memory crashes/BSODs.

---

## 3. Model Architecture Workarounds (`vision_3d.py`)
**WARNING FOR AI ASSISTANTS:** Do NOT refactor `JesterSequenceLSTM` back into a 3D CNN (`r3d_18`) or a standard `nn.LSTM`.
1. **No `Conv3d`:** DirectML fails on 5D volumetric tensors (`Conv3d`).
2. **No `nn.LSTM`:** DirectML lacks the `aten::_thnn_fused_lstm_cell` operator, causing a fatal CPU fallback crash during the forward pass.
3. **Current Hybrid Architecture:** 
   - Uses a 2D `resnet50` backbone (`ResNet50_Weights.DEFAULT`) applied frame-by-frame by flattening `[Batch, Channels, Time, H, W]` into `[Batch * Time, Channels, H, W]`.
   - `layer4` of ResNet50 is **unfrozen** (trained at `lr * 0.1`) to learn motion blur and hand shapes; earlier layers remain frozen.
   - Overridden `train()` method forces `self.cnn.eval()` so frozen BatchNorm statistics do not drift[cite: 2].
   - **Temporal Layer:** Uses `nn.Conv1d(2048, hidden_dim, kernel_size=3, padding=1)` assigned to the attribute `self.lstm` (kept for naming compatibility with `train.py`).

---

## 4. Planned Roadmap (Not Yet Fully in Code)
1. **Cloud & API Architecture (FastAPI + Docker + AWS):**
   - While training is happening locally via DirectML, the `cloud_backend` directory is structured to be containerized via **Docker** and deployed as a **FastAPI** inference microservice on **AWS** (e.g., EC2/ECS/S3 for weight storage), allowing lightweight client capture scripts to stream frame buffers or run hybrid local/cloud inference.
   - Keep backend code modular so device routing can cleanly toggle between `torch_directml` (local Windows dev) and standard `cuda`/`cpu` (when deployed in a Linux Docker container on AWS).
2. **Immediate Training Next Steps:**
   - Currently running a 1-epoch baseline on the 4-class Jester subset (9,710 clips across `Swiping Left`, `Swiping Right`, `No gesture`, `Doing other things`).
   - Need to evaluate the resulting Confusion Matrix and Precision/Recall report (`sklearn.metrics.classification_report`) to calculate exact inverse-frequency `class_weights` for `nn.CrossEntropyLoss(weight=class_weights)`.
   - Evaluate whether `AdaptiveAvgPool1d(1)` in `vision_3d.py` is averaging out temporal directionality (Left vs. Right trajectory) and whether flattening the temporal sequence directly into `self.fc` improves recall.
   - Build and wire the live OpenCV rolling-buffer webcam script (`webcam.py`) into the action dispatcher with confidence thresholding to prevent false triggers.

   ## Latest Epoch 1 Baseline Output
   --- Epoch 1 Completed | Loss: 1.2620 ---
                    precision    recall  f1-score   support

      Swiping Left       0.29      0.01      0.03      1762
     Swiping Right       0.31      0.01      0.03      1730
        No gesture       0.48      0.07      0.13      1844
Doing other things       0.46      0.97      0.63      4374

          accuracy                           0.46      9710
         macro avg       0.39      0.27      0.20      9710
      weighted avg       0.41      0.46      0.32      9710

Confusion Matrix:
 [[  24   25   47 1666]
 [  29   25   29 1647]
 [  13   10  138 1683]
 [  18   20   72 4264]]