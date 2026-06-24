#!/usr/bin/env python3
"""Download and extract supported person re-ID datasets."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "market1501": {
        "url": "https://drive.usercontent.google.com/download?id=0B8-rUzbwVRk0c054eEozWG9COHM&export=download&confirm=t",
        "archive": "Market-1501-v15.09.15.zip",
        "subdir": "market1501",
        "expected": "Market-1501-v15.09.15",
    },
    "msmt17": {
        "url": "https://huggingface.co/datasets/xianpeijie/MSMT17_V1/resolve/main/MSMT17_V1.zip",
        "archive": "MSMT17_V1.zip",
        "subdir": "msmt17",
        "expected": "MSMT17_V1",
    },
    "grid": {
        "url": "https://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/underground_reid.zip",
        "archive": "underground_reid.zip",
        "subdir": "grid",
        "expected": "underground_reid",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract Market-1501, MSMT17, and GRID into the data directory."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=[*DATASETS.keys(), "all"],
        default=None,
        help="Dataset(s) to download. Defaults to all.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Destination data directory. Defaults to ./data.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="Where zip files are stored. Defaults to <data-dir>/_downloads.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download and extract even if the expected dataset directory exists.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Only download zip files; do not extract them.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded zip files after successful extraction.",
    )
    return parser.parse_args()


def selected_datasets(names: list[str]) -> list[str]:
    if not names:
        return list(DATASETS.keys())
    if "all" in names:
        return list(DATASETS.keys())
    return names


def format_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown size"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def download_file(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"Using existing archive: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_destination = destination.with_suffix(destination.suffix + ".part")

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(request) as response, tmp_destination.open("wb") as out:
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else None
            downloaded = 0
            next_report = 0

            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)

                if total_bytes:
                    percent = int(downloaded * 100 / total_bytes)
                    if percent >= next_report:
                        print(
                            f"  {percent:3d}% ({format_size(downloaded)} / {format_size(total_bytes)})",
                            flush=True,
                        )
                        next_report += 10
                elif downloaded >= next_report:
                    print(f"  downloaded {format_size(downloaded)}", flush=True)
                    next_report += 100 * 1024 * 1024
    except urllib.error.URLError as exc:
        tmp_destination.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    tmp_destination.replace(destination)
    print(f"Saved archive: {destination}")


def ensure_zip(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise RuntimeError(
            f"{path} is not a valid zip file. For Google Drive, this usually means "
            "the response was an HTML confirmation page instead of the dataset."
        )


def extract_zip(archive_path: Path, destination_dir: Path, force: bool) -> None:
    if destination_dir.exists() and force:
        shutil.rmtree(destination_dir)

    destination_dir.mkdir(parents=True, exist_ok=True)
    ensure_zip(archive_path)
    print(f"Extracting {archive_path} -> {destination_dir}")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(destination_dir)


def process_dataset(name: str, args: argparse.Namespace) -> None:
    spec = DATASETS[name]
    dataset_root = args.data_dir / spec["subdir"]
    expected_dir = dataset_root / spec["expected"]
    download_dir = args.download_dir or args.data_dir / "_downloads"
    archive_path = download_dir / spec["archive"]

    if expected_dir.exists() and not args.force:
        print(f"Skipping {name}: found {expected_dir}")
        return

    download_file(spec["url"], archive_path, force=args.force)

    if args.no_extract:
        return

    extract_zip(archive_path, dataset_root, force=args.force)

    if not expected_dir.exists():
        raise RuntimeError(
            f"Extraction completed, but expected directory was not found: {expected_dir}"
        )

    if not args.keep_archives:
        archive_path.unlink()
        print(f"Removed archive: {archive_path}")
        try_remove_empty_dir(archive_path.parent)

    print(f"Ready: {expected_dir}")


def try_remove_empty_dir(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        return
    print(f"Removed empty download directory: {path}")


def main() -> int:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    if args.download_dir is not None:
        args.download_dir = args.download_dir.expanduser().resolve()

    for dataset in selected_datasets(args.datasets):
        process_dataset(dataset, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
