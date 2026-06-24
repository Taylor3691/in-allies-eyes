#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Adapt CAJ MSMT17_V1 layout to BoT MSMT17_V2 layout")
    parser.add_argument("--source", type=Path, default=Path("data/msmt17/MSMT17_V1"))
    parser.add_argument("--dest", type=Path, default=Path("data/msmt17/MSMT17_V2"))
    parser.add_argument("--copy", action="store_true", help="copy directories/files instead of creating symlinks")
    parser.add_argument("--force", action="store_true", help="replace existing broken links or files at destination")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def remove_target(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def target_is_ok(path):
    return path.exists()


def make_link_or_copy(src, dst, copy=False, force=False, dry_run=False):
    if target_is_ok(dst):
        print(f"exists: {dst}")
        return
    if dst.is_symlink() or dst.exists():
        if not force:
            raise RuntimeError(f"{dst} exists but is not valid; pass --force to replace it")
        print(f"remove: {dst}")
        if not dry_run:
            remove_target(dst)

    print(("copy" if copy else "link") + f": {dst} -> {src}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        rel_src = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel_src, target_is_directory=src.is_dir())


def main():
    args = parse_args()
    source = args.source.resolve()
    dest = args.dest

    required = ["train", "test", "list_train.txt", "list_val.txt", "list_query.txt", "list_gallery.txt"]
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise RuntimeError(f"Missing required MSMT17_V1 entries under {source}: {missing}")

    mapping = {
        "mask_train_v2": source / "train",
        "mask_test_v2": source / "test",
        "list_train.txt": source / "list_train.txt",
        "list_val.txt": source / "list_val.txt",
        "list_query.txt": source / "list_query.txt",
        "list_gallery.txt": source / "list_gallery.txt",
    }

    print(f"source: {source}")
    print(f"dest:   {dest}")
    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    for dst_name, src in mapping.items():
        make_link_or_copy(src, dest / dst_name, copy=args.copy, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
