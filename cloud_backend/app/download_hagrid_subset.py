#!/usr/bin/env python
"""
High-Performance HaGRID Subset Streamer / Extractor
===================================================
Streams balanced gesture samples directly from remote multi-gigabyte archives
using HTTP Range requests with high concurrency (32 workers) and fast resampling.
"""

import os
import io
import struct
import zlib
import time
import threading
import argparse
import requests
import zipfile
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

V2_URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/hagrid_v2_zip/"
MAIN_URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset/"

ARCHIVE_URLS = {
    "xsign": f"{V2_URL}xsign.zip",
    "grabbing": f"{V2_URL}grabbing.zip",
    "point": f"{V2_URL}point.zip",
    "stop": f"{MAIN_URL}stop.zip",
}

# Thread-local storage for HTTP sessions
_local = threading.local()

def get_thread_session():
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=2)
        _local.session.mount("https://", adapter)
    return _local.session


class HttpRangeReader(io.RawIOBase):
    """
    File-like wrapper that implements seek/tell/read via HTTP Range requests
    with block caching, enabling zipfile.ZipFile to read central directory tables.
    """
    def __init__(self, url, block_size=1024 * 1024):
        self.url = url
        self.block_size = block_size
        self.session = requests.Session()
        r = self.session.head(url, allow_redirects=True, timeout=15)
        self.size = int(r.headers.get("content-length", 0))
        self.pos = 0
        self.cache = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        else:
            raise ValueError("Invalid whence")
        return self.pos

    def _fetch_block(self, b_idx):
        if b_idx in self.cache:
            return self.cache[b_idx]
        start = b_idx * self.block_size
        end = min(self.size - 1, start + self.block_size - 1)
        r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=15)
        data = r.content
        if len(self.cache) > 30:
            self.cache.pop(next(iter(self.cache)))
        self.cache[b_idx] = data
        return data

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        req_len = len(b)
        n = 0
        while n < req_len and self.pos < self.size:
            b_idx = self.pos // self.block_size
            b_offset = self.pos % self.block_size
            block = self._fetch_block(b_idx)
            take = min(req_len - n, len(block) - b_offset)
            if take <= 0:
                break
            b[n:n + take] = block[b_offset:b_offset + take]
            self.pos += take
            n += take
        return n


def download_single_image(url, info, out_path, max_size=512):
    """
    Fetches, decompresses, resizes, and writes a single image from the remote archive.
    """
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        return True

    session = get_thread_session()
    est_len = 30 + 120 + info.compress_size + 128
    start = info.header_offset
    end = start + est_len

    for attempt in range(3):
        try:
            resp = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=15)
            if resp.status_code not in (200, 206):
                continue
            raw = resp.content

            # Parse local zip entry header
            magic, ver, flags, method, mtime, mdate, crc, csize, usize, fn_len, extra_len = struct.unpack(
                "<IHHHHHIIIHH", raw[:30]
            )
            if magic != 0x04034B50:
                continue

            payload_start = 30 + fn_len + extra_len
            comp_data = raw[payload_start:payload_start + info.compress_size]

            if method == 8:
                img_bytes = zlib.decompress(comp_data, -15)
            elif method == 0:
                img_bytes = comp_data
            else:
                continue

            # Fast decode, thumbnail, and save
            with Image.open(io.BytesIO(img_bytes)) as img:
                img = img.convert("RGB")
                img.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
                temp_path = f"{out_path}.tmp"
                img.save(temp_path, "JPEG", quality=85)
                os.replace(temp_path, out_path)
            return True
        except Exception:
            time.sleep(0.3 * (attempt + 1))

    return False


def extract_class_subset(class_name, archive_url, target_dir, num_samples=1200, max_workers=32):
    """
    Downloads num_samples for the given class into target_dir.
    """
    os.makedirs(target_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(target_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(existing_files) >= num_samples:
        print(f"[{class_name}] Already satisfied: {len(existing_files)} images present.", flush=True)
        return len(existing_files)

    needed = num_samples - len(existing_files)
    print(f"\n[{class_name}] Connecting to remote archive ({archive_url})...", flush=True)
    reader = HttpRangeReader(archive_url)
    with zipfile.ZipFile(reader) as z:
        all_infos = [
            info for info in z.infolist()
            if info.filename.lower().endswith((".jpg", ".jpeg", ".png")) and not info.filename.startswith("__MACOSX")
        ]
        print(f"[{class_name}] Found {len(all_infos)} total images in remote archive. Target: {num_samples} (need {needed} more)", flush=True)

        existing_set = set(existing_files)
        candidates = [info for info in all_infos if os.path.basename(info.filename) not in existing_set]
        to_download = candidates[:needed]

        print(f"[{class_name}] Downloading {len(to_download)} images with {max_workers} worker threads...", flush=True)
        success_count = 0
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_info = {}
            for info in to_download:
                fname = os.path.basename(info.filename)
                out_path = os.path.join(target_dir, fname)
                future = executor.submit(download_single_image, archive_url, info, out_path)
                future_to_info[future] = fname

            for idx, future in enumerate(as_completed(future_to_info), 1):
                if future.result():
                    success_count += 1
                if idx % 100 == 0 or idx == len(to_download):
                    elapsed = time.time() - start_time
                    fps = idx / max(1.0, elapsed)
                    print(f"[{class_name}] Progress: {idx}/{len(to_download)} ({success_count} succeeded, {fps:.1f} img/s)", flush=True)

    total_now = len([f for f in os.listdir(target_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"[{class_name}] Completed! Total images now: {total_now}\n", flush=True)
    return total_now


def main():
    parser = argparse.ArgumentParser(description="High-performance HaGRID subset downloader")
    parser.add_argument("--samples-per-class", type=int, default=1200,
                        help="Number of images per class (default: 1200)")
    parser.add_argument("--workers", type=int, default=32,
                        help="Number of concurrent download threads (default: 32)")
    parser.add_argument("--classes", nargs="+", default=["xsign", "grabbing", "point", "stop"],
                        help="Target classes to download (default: xsign grabbing point stop)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    base_data_dir = os.path.join(workspace_root, "data", "raw", "hagrid")

    print(f"=== HaGRID High-Speed Subset Downloader ===", flush=True)
    print(f"Base Directory: {base_data_dir}", flush=True)
    print(f"Classes: {args.classes}", flush=True)
    print(f"Samples per class: {args.samples_per_class}", flush=True)
    print(f"Worker threads: {args.workers}", flush=True)
    print("-" * 50, flush=True)

    for cls in args.classes:
        if cls not in ARCHIVE_URLS:
            print(f"Warning: No archive URL configured for class '{cls}', skipping.", flush=True)
            continue
        target_dir = os.path.join(base_data_dir, cls)
        extract_class_subset(
            class_name=cls,
            archive_url=ARCHIVE_URLS[cls],
            target_dir=target_dir,
            num_samples=args.samples_per_class,
            max_workers=args.workers
        )

    print("\nAll target classes downloaded successfully.", flush=True)


if __name__ == "__main__":
    main()
