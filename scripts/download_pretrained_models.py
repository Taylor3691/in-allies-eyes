#!/usr/bin/env python3
"""Download pretrained weights used by the CA-Jaccard experiments."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

RESNET50_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
RESNET50_FILE = "resnet50-11ad3fa6.pth"

GOOGLE_DRIVE_FILES = {
    "cc-market": ("CC_market1501_81.0.tar", "1Jza_z3tNv5cCtXan576qQsPfpB4tMeF6"),
    "caj-market": ("CC+CAJ_market1501_85.1.tar", "1Id7OplbL8ZpX6mw0wu847z7BoOuE2rub"),
    "market-bot": ("market_resnet50_model_120_rank1_945.pth", "14a2NLjFZMu1RmSKrljgytfhJpe1rCti7"),
    "cc-cuhk03": ("CC_cuhk03_6.4.tar", "1Dw0tfUzkgJ3uKqKsTKqwppSffDcScOZV"),
    "caj-cuhk03": ("CC+CAJ_cuhk03_61.6.tar", "1gUdgbYocUEVNOBQqaVKcF2PpbmS2stbM"),
    "cuhk03-bot": ("cuhk03_resnet50_model_120_rank1_608.pth", "1yBVUrmhUtmEDeh51-m0M772MYY6afzaD"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pretrained model weights into pretrained_models."
    )
    parser.add_argument(
        "models",
        nargs="*",
        choices=["all", "demo", "resnet50", *GOOGLE_DRIVE_FILES.keys()],
        default=None,
        help="Model weights to download. Defaults to all. Use 'demo' for the demo app weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "pretrained_models",
        help="Directory for final .pth files. Defaults to ./pretrained_models.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even if the expected file exists.",
    )
    return parser.parse_args()


def selected_models(names: list[str] | None) -> list[str]:
    if not names or "all" in names:
        return ["resnet50", *GOOGLE_DRIVE_FILES.keys()]
    selected = []
    for name in names:
        if name == "demo":
            selected.extend(GOOGLE_DRIVE_FILES.keys())
        else:
            selected.append(name)
    return selected


def google_drive_download_url(file_id: str) -> str:
    return (
        "https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm=t"
    )


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
        print(f"Using existing file: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_destination = destination.with_suffix(destination.suffix + ".part")

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"Downloading {destination.name}")
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
    print(f"Saved: {destination}")


def download_resnet50(output_dir: Path, force: bool) -> None:
    destination = output_dir / RESNET50_FILE
    if destination.exists() and not force:
        print(f"Skipping resnet50: found {destination}")
        return
    download_file(RESNET50_URL, destination, force=force)
    print(f"Ready: {destination}")


def download_google_drive_file(model_name: str, output_dir: Path, force: bool) -> None:
    filename, file_id = GOOGLE_DRIVE_FILES[model_name]
    destination = output_dir / filename
    if destination.exists() and not force:
        print(f"Skipping {model_name}: found {destination}")
        return
    download_file(google_drive_download_url(file_id), destination, force=force)
    print(f"Ready: {destination}")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for model in selected_models(args.models):
        if model == "resnet50":
            download_resnet50(output_dir, force=args.force)
        elif model in GOOGLE_DRIVE_FILES:
            download_google_drive_file(model, output_dir, force=args.force)
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
