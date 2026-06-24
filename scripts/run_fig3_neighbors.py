#!/usr/bin/env python3
import argparse
from pathlib import Path

from .experiment_utils import DATASET_DEFAULTS, append_csv, ensure_dataset, python_cmd, run_command, variant_flags


def parse_args():
    parser = argparse.ArgumentParser(description="Run Fig. 3 neighbor-analysis training commands")
    parser.add_argument("--dataset", choices=DATASET_DEFAULTS.keys(), default="market1501")
    parser.add_argument("--variants", nargs="+", default=["baseline", "ckrnns", "clqe", "caj"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--logs-root", default="logs/experiments/fig3_neighbors")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--jaccard-memory", choices=["auto", "dense", "sparse"], default="auto",
                        help="memory strategy for clustering Jaccard distance")
    parser.add_argument("--command-csv", default="results/fig3_commands.csv")
    return parser.parse_args()


def build_command(args, variant):
    defaults = DATASET_DEFAULTS[args.dataset]
    logs_dir = Path(args.logs_root) / args.dataset / variant
    analysis_file = logs_dir / "neighbor_analysis.csv"
    return python_cmd(
        "train_caj.py",
        "-d", args.dataset,
        "--data-dir", args.data_dir,
        "--logs-dir", logs_dir,
        "--eps", defaults["eps"],
        "--iters", defaults["iters"],
        "--height", defaults["height"],
        "--width", defaults["width"],
        "-b", args.batch_size,
        "-j", args.workers,
        "--jaccard-memory", args.jaccard_memory,
        "--neighbor-analysis",
        "--neighbor-analysis-file", analysis_file,
        *variant_flags(variant),
    )


def main():
    args = parse_args()
    ensure_dataset(args.dataset)

    for variant in args.variants:
        cmd = build_command(args, variant)
        append_csv(args.command_csv, {
            "experiment": "fig3",
            "scene": "clustering",
            "dataset": args.dataset,
            "variant": variant,
            "command": " ".join(map(str, cmd)),
        })
        run_command(cmd, dry_run=args.dry_run, gpu=args.gpu)


if __name__ == "__main__":
    main()
