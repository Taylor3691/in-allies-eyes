#!/usr/bin/env python3
"""Download pretrained weights used by the CA-Jaccard experiments."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

RESNET50_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
RESNET50_FILE = "resnet50-11ad3fa6.pth"

MARKET_BOT_FILE = "market_resnet50_model_120_rank1_945.pth"
MARKET_BOT_ARCHIVE = "market_resnet50_model_120_rank1_945.zip"
MARKET_BOT_PART_IDS = [
    "1cYQJ0_Kv-oPbeh0nwJPQUMoUHlOhZ1wb",
    "1AdVqIyKgH7wsny91ujeTAR62bidb9D3b",
    "1gHaSyepsiL0fFtWkygpSkwh-7eHkAuSP",
    "1YpocBA0H2o3Ss20L7ATpHhUP-74lcphT",
    "1bo5seGkArfZGDpFsuY_wCGN_6fWVZxDO",
    "18V08ix0qMjns_tokkgrlT-WDZKVJWHay",
    "1MOncX_Rz542ggbs0E2lqYLfMQMAcUwfx",
    "1VHnJRAalowWZO2IDmX4vcoVk56iSaZNf",
    "19yxxdgzlSMMDio2atKS_KJQQg3G6rXMT",
    "1pLcCjG13F4jnHyTFpK1L1w2PrJX8l7Mc",
    "1_y5f7HruaoTY_OA-bsqAeODV7mveEycM",
    "1Z82xJWmhcviZttgpEEv_Q8eJ2197sjxh",
    "1lOB_ZViXII_yHJx1AsNs1aOSZxs6Tk1p",
    "1hi_HHQwAILyTShcdOiEy97ibpLlaf3Z4",
    "1bF0BHuCJ_MJ42K2Tj38KwndttJbhOJaE",
    "1GZChLuO43dPvuPLkTqxA7kcmnCgdqH5P",
    "1l6qfaoiiIQYM_NyKtFiqzDs0Gk2vU_lC",
    "1UFNE0fwD5ERHakCnHXR0eCzWdFrzkt9N",
    "1O0ZKcKb-5B-5O8bHiWli87LITO9z5kQK",
    "1hhckvIwQlZRzYmFZfapUVMf4jbk7UT40",
    "1VFyk3lhDR0pkq3uNZy8fEwCAyRujpvGO",
    "1O-23PKovG1ZXcsDk8mKh21AjAQEaE059",
    "17p_44HsoqSPyEuoiN2xeiUuoIGYQ3HvQ",
    "1O-_jJ-c2Zmq1SdOSyVhE0-K-nG_vZL0N",
    "1KTtUYE9dKLNVaJboLjxDeBoWIEcqUw_S",
    "1iXjDcGMxbH_MZz3986iGmmyrHSUT3RIe",
    "1jmE0c9N00PHe1-JsviDxR2ggamSznBPt",
    "12aZ_27OF8HfnLN3ge8vPxvHwWK1zEMNP",
    "1eZPLvniF2xhSl9QvIQg39OU2VEUeQ7ie",
    "1wdf7uqh5_gxyctmaF-eM_udyZgKlxjHZ",
    "1UBPdfoX5ipniF-XYj7U9tGrxGx-mfvS9",
    "1tmlaMwB7D0naKie-R9BJK0_R-n6pIsj8",
    "1v0JoaUfesjo19JCgDRwZv9Q4S1Q40scP",
    "1u3N1ohy8tPKbBjujAO6LnHoAFqiGxEc5",
    "1OZCqI6EQrdYGN1x5gIUwWsY-jvQMgbC3",
    "1qS07fdPIIRWB5LqpHN6uFqaNcxaJgj5r",
    "1BVfd_AnMgur7TA63pY7Aemy7XtzxqhiO",
    "1QocapaIIwpuG6suaMJ9ykJCtIODucX9X",
    "1l7li_Cd4zXSkQhXcjCVPlyVaViKohASs",
    "1cSeVy_uLVSra8RtAcVBh7qRBx8cR5IDF",
    "1vFbbXabj-EQrXlRxzvqHEUjGnBWK-rFd",
    "1NaGcTDCcWVjGxtwDT35KhAwAwE3k_Re0",
    "1td0AEiZsIJj63mnsZMjiMZsae8BOi3pC",
    "1ri4A_yvGLflxWNYBnaf_cLNf0p9GewiX",
    "16Pmd80AALhqUMvlhRoF7ayAKs5B-vgUJ",
]

GOOGLE_DRIVE_FILES = {
    "cc-market": ("CC_market1501_81.0.tar", "1PQX3B7w37z-DX2_C5lZS4l3hZF00lDzK"),
    "caj-market": ("CC+CAJ_market1501_84.8.tar", "1YXNrdxrpKFa-0MlFaX_IhDLuTANt0rOj"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pretrained model weights into pretrained_models."
    )
    parser.add_argument(
        "models",
        nargs="*",
        choices=["all", "demo", "resnet50", "market-bot", "cc-market", "caj-market"],
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
        "--download-dir",
        type=Path,
        default=None,
        help="Directory for temporary archives. Defaults to <output-dir>/_downloads.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download and extract even if the expected .pth file exists.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded zip parts and the combined zip after extraction.",
    )
    return parser.parse_args()


def selected_models(names: list[str] | None) -> list[str]:
    if not names or "all" in names:
        return ["cc-market", "caj-market", "market-bot", "resnet50"]
    selected = []
    for name in names:
        if name == "demo":
            selected.extend(["cc-market", "caj-market", "market-bot"])
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


def download_market_bot(output_dir: Path, download_dir: Path, force: bool, keep_archives: bool) -> None:
    expected = output_dir / MARKET_BOT_FILE
    if expected.exists() and not force:
        print(f"Skipping market-bot: found {expected}")
        return

    parts_dir = download_dir / "market-bot"
    part_paths = []
    for index, file_id in enumerate(MARKET_BOT_PART_IDS, start=1):
        part_path = parts_dir / f"{MARKET_BOT_ARCHIVE}.{index:03d}"
        download_file(google_drive_download_url(file_id), part_path, force=force)
        part_paths.append(part_path)

    combined_zip = download_dir / MARKET_BOT_ARCHIVE
    combine_parts(part_paths, combined_zip, force=force)
    extract_zip(combined_zip, output_dir)
    ensure_expected_file(output_dir, MARKET_BOT_FILE)

    if not keep_archives:
        cleanup_market_bot_archives(parts_dir, combined_zip, download_dir)

    print(f"Ready: {expected}")


def download_google_drive_file(model_name: str, output_dir: Path, force: bool) -> None:
    filename, file_id = GOOGLE_DRIVE_FILES[model_name]
    destination = output_dir / filename
    if destination.exists() and not force:
        print(f"Skipping {model_name}: found {destination}")
        return
    download_file(google_drive_download_url(file_id), destination, force=force)
    print(f"Ready: {destination}")


def combine_parts(part_paths: list[Path], destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"Using existing combined archive: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_destination = destination.with_suffix(destination.suffix + ".part")
    print(f"Combining {len(part_paths)} parts -> {destination}")
    with tmp_destination.open("wb") as out:
        for part_path in part_paths:
            with part_path.open("rb") as part:
                shutil.copyfileobj(part, out, length=1024 * 1024)
    tmp_destination.replace(destination)


def extract_zip(archive_path: Path, output_dir: Path) -> None:
    if not zipfile.is_zipfile(archive_path):
        raise RuntimeError(
            f"{archive_path} is not a valid zip file. For Google Drive, this usually "
            "means one of the downloaded parts is an HTML confirmation/error page."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} -> {output_dir}")
    with zipfile.ZipFile(archive_path) as zf:
        safe_extract(zf, output_dir)


def safe_extract(zf: zipfile.ZipFile, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    for member in zf.infolist():
        target = (output_dir / member.filename).resolve()
        if output_root not in (target, *target.parents):
            raise RuntimeError(f"Refusing to extract unsafe path: {member.filename}")
    zf.extractall(output_dir)


def ensure_expected_file(output_dir: Path, filename: str) -> None:
    expected = output_dir / filename
    if expected.exists():
        return

    matches = list(output_dir.rglob(filename))
    if len(matches) == 1:
        matches[0].replace(expected)
        cleanup_empty_parents(matches[0].parent, stop_at=output_dir)
        return

    raise RuntimeError(f"Extraction completed, but expected file was not found: {expected}")


def cleanup_empty_parents(path: Path, stop_at: Path) -> None:
    stop_at = stop_at.resolve()
    current = path.resolve()
    while current != stop_at and stop_at in current.parents:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def cleanup_market_bot_archives(parts_dir: Path, combined_zip: Path, download_dir: Path) -> None:
    combined_zip.unlink(missing_ok=True)
    shutil.rmtree(parts_dir, ignore_errors=True)
    try_remove_empty_dir(download_dir)
    print("Removed market-bot zip archives.")


def try_remove_empty_dir(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        return


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    download_dir = (
        args.download_dir.expanduser().resolve()
        if args.download_dir is not None
        else output_dir / "_downloads"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for model in selected_models(args.models):
        if model == "resnet50":
            download_resnet50(output_dir, force=args.force)
        elif model == "market-bot":
            download_market_bot(
                output_dir,
                download_dir,
                force=args.force,
                keep_archives=args.keep_archives,
            )
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
