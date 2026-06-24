import csv
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_DEFAULTS = {
    "market1501": {"eps": 0.4, "iters": 200, "height": 256, "width": 128,
                   "batch_size": 256, "epochs": 50, "num_instances": 16},
    "msmt17": {"eps": 0.6, "iters": 400, "height": 256, "width": 128,
               "batch_size": 256, "epochs": 50, "num_instances": 16},
    "grid": {"eps": 0.3, "iters": 100, "height": 256, "width": 128,
             "batch_size": 64, "epochs": 30, "num_instances": 4},
}


def python_cmd(script, *args):
    return [sys.executable, str(REPO_ROOT / "scripts" / script), *map(str, args)]


def with_gpu_env(gpu):
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def print_command(cmd, cwd=None):
    prefix = f"(cd {cwd} && " if cwd else ""
    suffix = ")" if cwd else ""
    print(prefix + " ".join(map(str, cmd)) + suffix)


def run_command(cmd, dry_run=False, gpu=None, cwd=None):
    print_command(cmd, cwd=cwd)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=cwd, env=with_gpu_env(gpu), check=True).returncode


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)
        merged_fieldnames = existing_fieldnames[:]
        for fieldname in fieldnames:
            if fieldname not in merged_fieldnames:
                merged_fieldnames.append(fieldname)
        if merged_fieldnames != existing_fieldnames:
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=merged_fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)
        fieldnames = merged_fieldnames
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def extract_reid_metrics(log_path):
    log_path = Path(log_path)
    if not log_path.exists():
        return {}
    mean_ap = None
    rank1 = None
    rank5 = None
    lines = log_path.read_text(errors="ignore").splitlines()
    for idx, line in enumerate(lines):
        match = re.search(r"Mean AP:\s*([0-9.]+)%", line)
        if match:
            mean_ap = float(match.group(1))
            continue
        match = re.search(r"\bmAP:\s*([0-9.]+)%", line)
        if match:
            mean_ap = float(match.group(1))
            continue
        match = re.search(r"model mAP:\s*([0-9.]+)%", line)
        if match:
            mean_ap = float(match.group(1))
            continue
        if "top-1" in line:
            match = re.search(r"top-1\s+([0-9.]+)%", line)
            if match:
                rank1 = float(match.group(1))
        if "top-5" in line:
            match = re.search(r"top-5\s+([0-9.]+)%", line)
            if match:
                rank5 = float(match.group(1))
        match = re.search(r"CMC curve,\s*Rank-1\s*:?\s*([0-9.]+)%", line)
        if match:
            rank1 = float(match.group(1))
            continue
        match = re.search(r"CMC curve,\s*Rank-5\s*:?\s*([0-9.]+)%", line)
        if match:
            rank5 = float(match.group(1))
    metrics = {}
    if mean_ap is not None:
        metrics["mAP"] = mean_ap
    if rank1 is not None:
        metrics["Rank-1"] = rank1
    if rank5 is not None:
        metrics["Rank-5"] = rank5
    return metrics


def ensure_dataset(dataset):
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f"Unsupported dataset: {dataset}")


def variant_flags(variant):
    if variant == "baseline":
        return []
    if variant == "ckrnns":
        return ["--ckrnns"]
    if variant == "clqe":
        return ["--clqe", "--k2-intra", "2", "--k2-inter", "4"]
    if variant == "caj":
        return ["--ckrnns", "--clqe", "--k2-intra", "2", "--k2-inter", "4"]
    raise ValueError(f"Unsupported variant: {variant}")
