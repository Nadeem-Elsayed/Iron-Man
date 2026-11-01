import os
import hashlib
import json
import shutil
import time

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def backup_v1_weights():
    weights_dir = os.path.abspath("cloud_backend/weights")
    backup_dir = os.path.abspath("weights_v1.0.0_backup")
    os.makedirs(backup_dir, exist_ok=True)

    production_models = [
        "hagrid_resnet50_finetuned.pth",
        "jester_lstm_spatial_tuned.pth"
    ]

    manifest = {
        "version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "production_weights": {},
        "all_checkpoints": {}
    }

    files = sorted([f for f in os.listdir(weights_dir) if f.endswith(".pth")])
    print(f"Found {len(files)} model weight files in {weights_dir}")

    for filename in files:
        filepath = os.path.join(weights_dir, filename)
        size_bytes = os.path.getsize(filepath)
        size_mb = round(size_bytes / (1024 * 1024), 2)
        print(f"Hashing {filename} ({size_mb} MB)...")
        sha256 = compute_sha256(filepath)

        entry = {
            "size_bytes": size_bytes,
            "size_mb": size_mb,
            "sha256": sha256,
            "last_modified": time.ctime(os.path.getmtime(filepath))
        }
        manifest["all_checkpoints"][filename] = entry

        if filename in production_models:
            manifest["production_weights"][filename] = entry
            dest = os.path.join(backup_dir, filename)
            if not os.path.exists(dest) or os.path.getsize(dest) != size_bytes:
                print(f"  --> Backing up {filename} to {dest}...")
                shutil.copy2(filepath, dest)
            else:
                print(f"  --> Backup of {filename} already up to date.")

    manifest_path = os.path.join(backup_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Also save a copy inside cloud_backend/weights
    with open(os.path.join(weights_dir, "WEIGHTS_v1.0.0_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSuccessfully generated manifest at: {manifest_path}")
    print(f"Backed up production weights to: {backup_dir}")

if __name__ == "__main__":
    backup_v1_weights()
