import os
import hashlib
import json
import shutil
import time
import argparse

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def backup_weights(version="2.0.0"):
    weights_dir = os.path.abspath("cloud_backend/weights")
    backup_dir = os.path.abspath(f"weights_v{version}_backup")
    os.makedirs(backup_dir, exist_ok=True)

    if version.startswith("2."):
        production_models = [
            "hagrid_resnet50_v2.pth",
            "hagrid_resnet50_finetuned.pth",
            "jester_lstm_spatial_tuned.pth"
        ]
    else:
        production_models = [
            "hagrid_resnet50_finetuned.pth",
            "jester_lstm_spatial_tuned.pth"
        ]

    manifest = {
        "version": version,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "production_weights": {},
        "all_checkpoints": {}
    }

    files = sorted([f for f in os.listdir(weights_dir) if f.endswith(".pth")])
    print(f"Found {len(files)} model weight files in {weights_dir} for Release {version}")

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

    with open(os.path.join(weights_dir, f"WEIGHTS_v{version}_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSuccessfully generated manifest at: {manifest_path}")
    print(f"Backed up production weights to: {backup_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="2.0.0")
    args = parser.parse_args()
    backup_weights(version=args.version)
