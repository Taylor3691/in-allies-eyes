#!/usr/bin/env python3
import argparse
from pathlib import Path

from .experiment_utils import (
    DATASET_DEFAULTS,
    append_csv,
    ensure_dataset,
    extract_reid_metrics,
    python_cmd,
    run_command,
    variant_flags,
)


CLUSTER_VARIANTS = ["baseline", "ckrnns", "clqe", "caj"]
RERANK_VARIANTS = ["baseline", "ckrnns", "clqe", "caj"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Tab. 3 ablation commands")
    parser.add_argument("--dataset", choices=DATASET_DEFAULTS.keys(), default="market1501")
    parser.add_argument("--scene", choices=["clustering", "reranking", "both"], default="clustering")
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--logs-root", default="logs/experiments/tab3")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cluster-checkpoint", default="",
                        help=("CAJ checkpoint for clustering scene evaluation. "
                              "If set, test.py is run instead of train_caj.py; "
                              "supports {dataset} and {variant} placeholders."))
    parser.add_argument("--bot-checkpoint", default="", help="BoT checkpoint for re-ranking scene")
    parser.add_argument("--bot-neck-feat", choices=["after", "before"], default="after",
                        help="BoT BNNeck feature used by test.py for re-ranking")
    parser.add_argument("--bot-no-feat-norm", action="store_true",
                        help="disable L2 normalization of BoT test features")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--jaccard-memory", choices=["auto", "dense", "sparse"], default="auto",
                        help="memory strategy for clustering Jaccard distance")
    parser.add_argument("--command-csv", default="results/tab3_commands.csv")
    parser.add_argument("--results-csv", default="results/tab3_results.csv")
    return parser.parse_args()


def resolve_cluster_checkpoint(args, variant):
    if not args.cluster_checkpoint:
        return ""
    return args.cluster_checkpoint.format(dataset=args.dataset, variant=variant)


def clustering_command(args, variant):
    defaults = DATASET_DEFAULTS[args.dataset]
    logs_dir = Path(args.logs_root) / "clustering" / args.dataset / variant
    checkpoint = resolve_cluster_checkpoint(args, variant)
    if checkpoint:
        return python_cmd(
            "test.py",
            "-d", args.dataset,
            "--data-dir", args.data_dir,
            "--logs-dir", logs_dir,
            "--resume", checkpoint,
            "--checkpoint-format", "caj",
            "--height", defaults["height"],
            "--width", defaults["width"],
            "-b", args.batch_size,
            "-j", args.workers,
        ), logs_dir, logs_dir / "log_test.txt"

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
        *variant_flags(variant),
    ), logs_dir, logs_dir / "log.txt"


def rerank_command(args, variant):
    defaults = DATASET_DEFAULTS[args.dataset]
    logs_dir = Path(args.logs_root) / "reranking" / args.dataset / variant
    cmd = python_cmd(
        "test.py",
        "-d", args.dataset,
        "--data-dir", args.data_dir,
        "--logs-dir", logs_dir,
        "--resume", args.bot_checkpoint,
        "--checkpoint-format", "bot",
        "--bot-neck-feat", args.bot_neck_feat,
        "--height", defaults["height"],
        "--width", defaults["width"],
        "-b", args.batch_size,
        "-j", args.workers,
    )
    if args.bot_no_feat_norm:
        cmd.append("--bot-no-feat-norm")
    if variant != "baseline":
        cmd.append("--rerank")
    if variant == "ckrnns":
        cmd.extend(["--ckrnns"])
    elif variant == "clqe":
        cmd.extend(["--clqe", "--k2-intra", "2", "--k2-inter", "4"])
    elif variant == "caj":
        cmd.extend(["--ckrnns", "--clqe", "--k2-intra", "2", "--k2-inter", "4"])
    elif variant != "baseline":
        raise ValueError(f"Unsupported re-ranking variant: {variant}")
    return cmd, logs_dir


def record_command(args, scene, variant, cmd):
    append_csv(args.command_csv, {
        "experiment": "tab3",
        "scene": scene,
        "dataset": args.dataset,
        "variant": variant,
        "command": " ".join(map(str, cmd)),
    })


def record_result(args, scene, variant, log_path):
    metrics = extract_reid_metrics(log_path)
    if not metrics:
        return
    row = {
        "experiment": "tab3",
        "scene": scene,
        "dataset": args.dataset,
        "variant": variant,
    }
    row.update(metrics)
    append_csv(args.results_csv, row)


def main():
    args = parse_args()
    ensure_dataset(args.dataset)

    scenes = ["clustering", "reranking"] if args.scene == "both" else [args.scene]
    for scene in scenes:
        if scene == "clustering":
            variants = args.variants or CLUSTER_VARIANTS
            for variant in variants:
                if variant not in CLUSTER_VARIANTS:
                    raise ValueError(f"Unsupported clustering variant: {variant}")
                cmd, logs_dir, log_path = clustering_command(args, variant)
                record_command(args, scene, variant, cmd)
                run_command(cmd, dry_run=args.dry_run, gpu=args.gpu)
                if not args.dry_run:
                    record_result(args, scene, variant, log_path)
        else:
            if not args.bot_checkpoint:
                raise ValueError("--bot-checkpoint is required for re-ranking scene")
            variants = args.variants or RERANK_VARIANTS
            for variant in variants:
                if variant not in RERANK_VARIANTS:
                    raise ValueError(f"Unsupported re-ranking variant: {variant}")
                cmd, logs_dir = rerank_command(args, variant)
                record_command(args, scene, variant, cmd)
                run_command(cmd, dry_run=args.dry_run, gpu=args.gpu)
                if not args.dry_run:
                    record_result(args, scene, variant, logs_dir / "log_test.txt")


if __name__ == "__main__":
    main()
