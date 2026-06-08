import concurrent.futures
import json
import os
import threading
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "full_eval_downloads"
ARCHIVE_DIR = BASE_DIR / "archives"
POPE_DIR = BASE_DIR / "extracted" / "pope" / "coco"
STATUS_PATH = BASE_DIR / "download_status.json"

WORKERS = int(os.environ.get("EVAL_DOWNLOAD_WORKERS", "8"))
CHUNK_SIZE = 1024 * 1024
MULTIPART_MIN_BYTES = 16 * 1024 * 1024
MAX_RETRIES = int(os.environ.get("EVAL_DOWNLOAD_RETRIES", "20"))


FILES = [
    {
        "name": "gqa_eval.zip",
        "url": "https://nlp.stanford.edu/data/gqa/eval.zip",
        "path": ARCHIVE_DIR / "gqa_eval.zip",
        "size": 821_885_879,
    },
    {
        "name": "pope_coco_adversarial.json",
        "url": "https://raw.githubusercontent.com/AoiDragon/POPE/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_adversarial.json",
        "path": POPE_DIR / "coco_pope_adversarial.json",
        "size": 370_459,
    },
    {
        "name": "pope_coco_popular.json",
        "url": "https://raw.githubusercontent.com/AoiDragon/POPE/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_popular.json",
        "path": POPE_DIR / "coco_pope_popular.json",
        "size": 370_234,
    },
    {
        "name": "pope_coco_random.json",
        "url": "https://raw.githubusercontent.com/AoiDragon/POPE/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_random.json",
        "path": POPE_DIR / "coco_pope_random.json",
        "size": 360_212,
    },
    {
        "name": "coco_val2014.zip",
        "url": "http://images.cocodataset.org/zips/val2014.zip",
        "path": ARCHIVE_DIR / "coco_val2014.zip",
        "size": 6_645_013_297,
    },
    {
        "name": "gqa_images.zip",
        "url": "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
        "path": ARCHIVE_DIR / "gqa_images.zip",
        "size": 21_817_965_542,
    },
]


def fmt_seconds(seconds):
    if seconds is None:
        return None
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def part_ranges(size, workers):
    part_count = max(1, min(workers, size))
    step = (size + part_count - 1) // part_count
    ranges = []
    for index in range(part_count):
        start = index * step
        if start >= size:
            break
        end = min(start + step - 1, size - 1)
        ranges.append((index, start, end))
    return ranges


def part_path(path, index):
    return path.with_name(f"{path.name}.part{index:02d}")


def item_bytes(item):
    path = item["path"]
    expected = item["size"]
    if path.exists() and path.stat().st_size == expected:
        return expected

    parts = part_ranges(expected, WORKERS)
    part_total = 0
    for index, start, end in parts:
        pp = part_path(path, index)
        if pp.exists():
            part_total += min(pp.stat().st_size, end - start + 1)
    if part_total:
        return part_total

    if path.exists():
        return min(path.stat().st_size, expected)
    return 0


def current_total_bytes():
    return sum(item_bytes(item) for item in FILES)


def write_status(status):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    tmp.replace(STATUS_PATH)


def ensure_not_partial_final(path, expected):
    if not path.exists():
        return
    actual = path.stat().st_size
    if actual == expected:
        return
    backup = path.with_name(f"{path.name}.single.partial")
    if backup.exists():
        backup.unlink()
    path.replace(backup)


def restore_single_partial(path):
    backup = path.with_name(f"{path.name}.single.partial")
    if not path.exists() and backup.exists():
        backup.replace(path)


def download_range(url, output_path, start, end):
    expected = end - start + 1
    existing = output_path.stat().st_size if output_path.exists() else 0
    if existing == expected:
        return
    if existing > expected:
        output_path.unlink()
        existing = 0

    request = urllib.request.Request(url)
    request.add_header("Range", f"bytes={start + existing}-{end}")

    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError(f"Range request failed for {output_path.name}: HTTP {response.status}")
        with open(output_path, "ab") as output:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)

    actual = output_path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"{output_path.name} size mismatch: got {actual}, expected {expected}")


def monitor_progress(item, stop_event, file_index, file_count, total_size):
    last_time = time.time()
    last_total = current_total_bytes()

    while not stop_event.wait(2):
        now = time.time()
        total_done = current_total_bytes()
        file_done = item_bytes(item)
        delta_bytes = total_done - last_total
        delta_time = max(now - last_time, 1e-6)
        speed = delta_bytes / delta_time
        remaining = max(total_size - total_done, 0)
        eta = remaining / speed if speed > 0 else None

        write_status({
            "state": "downloading",
            "file_index": file_index,
            "file_count": file_count,
            "current_file": item["name"],
            "current_bytes": file_done,
            "current_size": item["size"],
            "current_percent": 100 * file_done / item["size"],
            "total_bytes": total_done,
            "total_size": total_size,
            "total_percent": 100 * total_done / total_size,
            "speed_mib_s": speed / (1024 * 1024),
            "eta_seconds": eta,
            "eta": fmt_seconds(eta),
            "workers": WORKERS,
            "updated_at": now,
        })
        last_time = now
        last_total = total_done


def assemble_parts(item):
    path = item["path"]
    expected = item["size"]
    parts = part_ranges(expected, WORKERS)
    tmp = path.with_name(f"{path.name}.assembling")
    if tmp.exists():
        tmp.unlink()

    with open(tmp, "wb") as output:
        for index, start, end in parts:
            pp = part_path(path, index)
            part_size = end - start + 1
            actual = pp.stat().st_size if pp.exists() else -1
            if actual != part_size:
                raise RuntimeError(f"{pp.name} size mismatch: got {actual}, expected {part_size}")
            with open(pp, "rb") as source:
                while True:
                    chunk = source.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)

    if tmp.stat().st_size != expected:
        raise RuntimeError(f"{tmp.name} size mismatch: got {tmp.stat().st_size}, expected {expected}")

    tmp.replace(path)
    for index, _, _ in parts:
        part_path(path, index).unlink(missing_ok=True)


def download_multipart(item, file_index, file_count, total_size):
    path = item["path"]
    expected = item["size"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == expected:
        return

    ensure_not_partial_final(path, expected)
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=monitor_progress,
        args=(item, stop_event, file_index, file_count, total_size),
        daemon=True,
    )
    monitor.start()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = []
            for index, start, end in part_ranges(expected, WORKERS):
                futures.append(executor.submit(
                    download_range,
                    item["url"],
                    part_path(path, index),
                    start,
                    end,
                ))
            for future in concurrent.futures.as_completed(futures):
                future.result()
    finally:
        stop_event.set()
        monitor.join(timeout=5)

    assemble_parts(item)


def download_simple(item, file_index, file_count, total_size):
    path = item["path"]
    expected = item["size"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == expected:
        return
    if path.exists() and path.stat().st_size > expected:
        path.unlink()

    last_time = time.time()
    last_total = current_total_bytes()
    attempts = 0

    while True:
        existing = path.stat().st_size if path.exists() else 0
        if existing == expected:
            return

        request = urllib.request.Request(item["url"])
        if existing:
            request.add_header("Range", f"bytes={existing}-")

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                mode = "ab" if existing and response.status == 206 else "wb"
                if existing and response.status != 206:
                    existing = 0
                with open(path, mode) as output:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        output.write(chunk)
                        now = time.time()
                        if now - last_time < 2:
                            continue
                        total_done = current_total_bytes()
                        file_done = item_bytes(item)
                        speed = (total_done - last_total) / max(now - last_time, 1e-6)
                        remaining = max(total_size - total_done, 0)
                        eta = remaining / speed if speed > 0 else None
                        write_status({
                            "state": "downloading",
                            "file_index": file_index,
                            "file_count": file_count,
                            "current_file": item["name"],
                            "current_bytes": file_done,
                            "current_size": expected,
                            "current_percent": 100 * file_done / expected,
                            "total_bytes": total_done,
                            "total_size": total_size,
                            "total_percent": 100 * total_done / total_size,
                            "speed_mib_s": speed / (1024 * 1024),
                            "eta_seconds": eta,
                            "eta": fmt_seconds(eta),
                            "workers": 1,
                            "retry_attempt": attempts,
                            "updated_at": now,
                        })
                        last_time = now
                        last_total = total_done
            attempts = 0
        except Exception as exc:
            attempts += 1
            if attempts > MAX_RETRIES:
                raise
            sleep_seconds = min(60, 5 * attempts)
            write_status({
                "state": "retrying",
                "file_index": file_index,
                "file_count": file_count,
                "current_file": item["name"],
                "current_bytes": item_bytes(item),
                "current_size": expected,
                "current_percent": 100 * item_bytes(item) / expected,
                "total_bytes": current_total_bytes(),
                "total_size": total_size,
                "total_percent": 100 * current_total_bytes() / total_size,
                "workers": 1,
                "retry_attempt": attempts,
                "retry_sleep_seconds": sleep_seconds,
                "last_error": str(exc),
                "updated_at": time.time(),
            })
            time.sleep(sleep_seconds)


def download_file(item, file_index, file_count, total_size):
    if WORKERS <= 1:
        restore_single_partial(item["path"])
        download_simple(item, file_index, file_count, total_size)
    elif item["size"] >= MULTIPART_MIN_BYTES:
        download_multipart(item, file_index, file_count, total_size)
    else:
        download_simple(item, file_index, file_count, total_size)


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    POPE_DIR.mkdir(parents=True, exist_ok=True)

    total_size = sum(item["size"] for item in FILES)
    start = time.time()

    write_status({
        "state": "starting",
        "total_size": total_size,
        "total_percent": 100 * current_total_bytes() / total_size,
        "workers": WORKERS,
        "updated_at": start,
    })

    for index, item in enumerate(FILES, start=1):
        write_status({
            "state": "starting_file",
            "file_index": index,
            "file_count": len(FILES),
            "current_file": item["name"],
            "current_bytes": item_bytes(item),
            "current_size": item["size"],
            "current_percent": 100 * item_bytes(item) / item["size"],
            "total_size": total_size,
            "total_percent": 100 * current_total_bytes() / total_size,
            "workers": WORKERS,
            "updated_at": time.time(),
        })
        download_file(item, index, len(FILES), total_size)

    write_status({
        "state": "complete",
        "total_bytes": current_total_bytes(),
        "total_size": total_size,
        "total_percent": 100.0,
        "elapsed_seconds": time.time() - start,
        "elapsed": fmt_seconds(time.time() - start),
        "updated_at": time.time(),
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_status({
            "state": "error",
            "error": str(exc),
            "total_bytes": current_total_bytes(),
            "updated_at": time.time(),
        })
        raise
