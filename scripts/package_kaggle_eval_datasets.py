import json
import shutil
import time
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = ROOT / "data" / "full_eval_downloads"
ARCHIVE_DIR = DOWNLOAD_DIR / "archives"
POPE_SOURCE_DIR = DOWNLOAD_DIR / "extracted" / "pope" / "coco"

PACKAGE_DIR = ROOT / "data" / "kaggle_datasets"
GQA_DIR = PACKAGE_DIR / "GQA"
POPE_DIR = PACKAGE_DIR / "POPE"
STATUS_PATH = PACKAGE_DIR / "package_status.json"

GQA_ZIP = PACKAGE_DIR / "GQA.zip"
POPE_ZIP = PACKAGE_DIR / "POPE.zip"


def fmt_seconds(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def write_status(**status):
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    status["updated_at"] = time.time()
    tmp = STATUS_PATH.with_suffix(".json.tmp")
    payload = json.dumps(status, indent=2)
    for attempt in range(10):
        try:
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(STATUS_PATH)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(path)


def extract_zip(zip_path, dest_dir, marker_name, label):
    marker = dest_dir / marker_name
    if marker.exists():
        write_status(state="skipped_extract", task=label, detail="marker exists")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        members = archive.infolist()
        total = len(members)
        start = time.time()
        for index, member in enumerate(members, start=1):
            target = dest_dir / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.exists() or target.stat().st_size != member.file_size:
                archive.extract(member, dest_dir)
            if index == total or index % 1000 == 0:
                elapsed = time.time() - start
                rate = index / elapsed if elapsed > 0 else 0
                eta = (total - index) / rate if rate > 0 else 0
                write_status(
                    state="extracting",
                    task=label,
                    current=index,
                    total=total,
                    percent=100 * index / total,
                    eta=fmt_seconds(eta),
                )

    marker.write_text(f"Extracted from {zip_path.name}\n", encoding="utf-8")


def copy_pope_annotations():
    annotation_dir = POPE_DIR / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        POPE_SOURCE_DIR / "coco_pope_adversarial.json",
        POPE_SOURCE_DIR / "coco_pope_popular.json",
        POPE_SOURCE_DIR / "coco_pope_random.json",
    ]
    for source in sources:
        require_file(source)
        shutil.copy2(source, annotation_dir / source.name)
    write_status(state="copied", task="POPE annotations", current=len(sources), total=len(sources), percent=100.0)


def prune_gqa_eval_training_choices():
    train_choices = GQA_DIR / "eval" / "train_choices"
    if train_choices.exists():
        shutil.rmtree(train_choices)
    write_status(state="pruned", task="GQA training choices", detail="removed eval/train_choices")


def iter_files(root):
    for path in root.rglob("*"):
        if path.is_file() and path.name != "package_status.json":
            yield path


def zip_folder(folder, output_zip, label):
    files = list(iter_files(folder))
    total = len(files)
    if output_zip.exists():
        output_zip.unlink()

    start = time.time()
    with ZipFile(output_zip, "w", compression=ZIP_STORED, allowZip64=True) as archive:
        for index, path in enumerate(files, start=1):
            archive.write(path, path.relative_to(PACKAGE_DIR))
            if index == total or index % 1000 == 0:
                elapsed = time.time() - start
                rate = index / elapsed if elapsed > 0 else 0
                eta = (total - index) / rate if rate > 0 else 0
                write_status(
                    state="zipping",
                    task=label,
                    current=index,
                    total=total,
                    percent=100 * index / total if total else 100.0,
                    eta=fmt_seconds(eta),
                    output=str(output_zip),
                )


def main():
    for path in [
        ARCHIVE_DIR / "gqa_eval.zip",
        ARCHIVE_DIR / "gqa_images.zip",
        ARCHIVE_DIR / "coco_val2014.zip",
        POPE_SOURCE_DIR / "coco_pope_adversarial.json",
        POPE_SOURCE_DIR / "coco_pope_popular.json",
        POPE_SOURCE_DIR / "coco_pope_random.json",
    ]:
        require_file(path)

    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    write_status(state="starting", task="package Kaggle eval datasets")

    extract_zip(ARCHIVE_DIR / "gqa_eval.zip", GQA_DIR / "eval", ".gqa_eval_extract_complete", "GQA eval")
    prune_gqa_eval_training_choices()
    extract_zip(ARCHIVE_DIR / "gqa_images.zip", GQA_DIR, ".gqa_images_extract_complete", "GQA images")

    copy_pope_annotations()
    extract_zip(ARCHIVE_DIR / "coco_val2014.zip", POPE_DIR, ".coco_val2014_extract_complete", "POPE COCO val2014")

    zip_folder(GQA_DIR, GQA_ZIP, "GQA.zip")
    zip_folder(POPE_DIR, POPE_ZIP, "POPE.zip")

    write_status(
        state="complete",
        task="package Kaggle eval datasets",
        gqa_zip=str(GQA_ZIP),
        pope_zip=str(POPE_ZIP),
        gqa_zip_bytes=GQA_ZIP.stat().st_size,
        pope_zip_bytes=POPE_ZIP.stat().st_size,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_status(state="error", task="package Kaggle eval datasets", error=str(exc))
        raise
