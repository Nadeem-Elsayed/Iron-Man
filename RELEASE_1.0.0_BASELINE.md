# STARK OS // IronMan - Release 1.0.0 Baseline Archive

**Release Version:** `v1.0.0`  
**Date:** November 1, 2026  
**Git Baseline Commit:** `cea7e9e` (and tagged `v1.0.0`)  
**Architecture:** Decoupled Edge Client (PyQt/OpenCV Win32) + Serverless Cloud Microservice (FastAPI on AWS Lambda)

---

## 1. Production Model Weights & Checkpoints

All 26 model weight checkpoints are inventoried in `cloud_backend/weights/` and cryptographic hashes are tracked in `weights_v1.0.0_backup/manifest.json`.

### Primary Production Checkpoints (Backed up in `weights_v1.0.0_backup/`)

| Model Stream | Filename | Size (MB) | SHA-256 Hash | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **2D Static Gesture** | `hagrid_resnet50_finetuned.pth` | 136.04 MB | `dac639f04e696f09cdb353b41ba4e4327f1fe8d61a741c17713051af6aa98115` | Production ResNet-50 classifier (5 HaGRID classes), deployed in AWS Lambda microservice container. |
| **3D Dynamic Motion** | `jester_lstm_spatial_tuned.pth` | 163.42 MB | `64183f508d98f732316b4938ebb093828f5e2016b78a3206a37864d44ce7cda1` | Spatial-tuned 1D-temporal CNN classifier (4 Jester classes). |

### Cloud S3 Backup
- **S3 Bucket URI:** `s3://ironman-models-323443471817/hagrid_resnet50_finetuned.pth`
- **Region:** `us-east-2`
- **Container Registry:** `323443471817.dkr.ecr.us-east-2.amazonaws.com/gesture-control:latest`
- **AWS Lambda Production Endpoint:** `https://nxebhxb63n4xfilodiwddn4jgq0rrcij.lambda-url.us-east-2.on.aws/predict`

---

## 2. Gesture Actions Mapping (v1.0.0)

### 2D Static Stream (HaGRID 5 Classes)
- `heart` / `hand_heart`: Launch or switch to Steam application (`steam://open/main`)
- `thumb_index2`: Instant bidirectional Show/Hide Desktop (`win + d`) keeping HUD topmost
- `take_photo` / `take_picture`: Instant full-screen screenshot saved to `Pictures/Screenshots`
- `mute`: Master volume mute/unmute toggle (`volumemute`)
- `no_gesture`: Passive control state (no action triggered)

### 3D Dynamic Stream (20BN-Jester 4 Classes)
- `swipe_left`: Previous Window (`alt + shift + tab`)
- `swipe_right`: Next Window (`alt + tab`)
- `no_gesture`: Static control state (no action)
- `doing_other_things`: Dynamic non-gesture movement state (no action)

---

## 3. Client Binary Artifacts (v1.0.0)

- **Standalone Directory:** `dist/IronManClient/`
- **Main Executable:** `dist/IronManClient/IronManClient.exe`
- **Launcher Batch File:** `dist/IronManClient/START_IRONMAN.bat`
- **Zip Distribution Archive:** `dist/IronManClient-v1.0.0-Windows.zip` (75.9 MB)
