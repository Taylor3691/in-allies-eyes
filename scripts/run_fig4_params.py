#!/usr/bin/env python3
import argparse
import sys
import os.path as osp
from pathlib import Path

# Add scripts directory to path to enable direct import of experiment_utils
scripts_dir = osp.dirname(osp.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from experiment_utils import (
    DATASET_DEFAULTS,
    append_csv,
    ensure_dataset,
    extract_reid_metrics,
    python_cmd,
    run_command,
)


def parse_int_list(value):
    return [int(part) for part in value.split(",") if part]


def parse_k2_pairs(value):
    pairs = []
    for item in value.split(","):
        left, right = item.split("/")
        pairs.append((int(left), int(right)))
    return pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Run Fig. 4 parameter-analysis commands")
    parser.add_argument("--dataset", choices=DATASET_DEFAULTS.keys(), default="market1501")
    parser.add_argument("--scene", choices=["clustering", "reranking", "both"], default="clustering")
    parser.add_argument("--sweep", choices=["k1-intra", "k1-inter", "k2", "all"], default="all")
    parser.add_argument("--k1-intra-values", default="1,5,10,15,20,25,40")
    parser.add_argument("--k1-inter-values", default="5,10,15,20,25,30,40")
    parser.add_argument("--k2-pairs", default="1/5,2/4,3/3,4/2,5/1")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--logs-root", default="logs/experiments/fig4_params")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bot-checkpoint", default="", help="BoT checkpoint for re-ranking scene")
    parser.add_argument("--bot-neck-feat", choices=["after", "before"], default="after",
                        help="BoT BNNeck feature used by test.py for re-ranking")
    parser.add_argument("--bot-no-feat-norm", action="store_true",
                        help="disable L2 normalization of BoT test features")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: per-dataset from DATASET_DEFAULTS)")
    parser.add_argument(
        "--num-instances",
        type=int,
        default=None,
        help=(
            "Override train_caj.py --num-instances for clustering scene "
            "(default: per-dataset from DATASET_DEFAULTS)."
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--jaccard-memory", choices=["auto", "dense", "sparse"], default="auto",
                        help="memory strategy for clustering Jaccard distance")
    parser.add_argument("--epochs", type=int, default=None, help="Override train epochs for clustering scene")
    parser.add_argument("--iters", type=int, default=None, help="Override train iters per epoch for clustering scene")
    parser.add_argument("--command-csv", default="results/fig4_commands.csv")
    parser.add_argument("--results-csv", default="results/fig4_results.csv")
    return parser.parse_args()


def clustering_command(args, sweep_name, label, extra_flags):
    defaults = DATASET_DEFAULTS[args.dataset]
    logs_dir = Path(args.logs_root) / "clustering" / args.dataset / sweep_name / label
    iters = defaults["iters"] if args.iters is None else args.iters
    batch_size = args.batch_size if args.batch_size is not None else defaults["batch_size"]
    epochs = args.epochs if args.epochs is not None else defaults["epochs"]
    num_instances = args.num_instances if args.num_instances is not None else defaults["num_instances"]
    return python_cmd(
        "train_caj.py",
        "-d", args.dataset,
        "--data-dir", args.data_dir,
        "--logs-dir", logs_dir,
        "--eps", defaults["eps"],
        "--iters", iters,
        "--height", defaults["height"],
        "--width", defaults["width"],
        "-b", batch_size,
        "--epochs", epochs,
        "--num-instances", num_instances,
        "-j", args.workers,
        "--jaccard-memory", args.jaccard_memory,
        "--ckrnns", "--clqe",
        *extra_flags,
    ), logs_dir


def rerank_command(args, sweep_name, label, extra_flags):
    defaults = DATASET_DEFAULTS[args.dataset]
    logs_dir = Path(args.logs_root) / "reranking" / args.dataset / sweep_name / label
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
        "--rerank", "--ckrnns", "--clqe",
        *extra_flags,
    )
    if args.bot_no_feat_norm:
        cmd.append("--bot-no-feat-norm")
    return cmd, logs_dir


def sweep_items(args):
    selected = {"k1-intra", "k1-inter", "k2"} if args.sweep == "all" else {args.sweep}
    items = []
    if "k1-intra" in selected:
        for value in parse_int_list(args.k1_intra_values):
            items.append(("k1-intra", str(value), ["--k1-intra", value, "--k1-inter", 20, "--k2-intra", 2, "--k2-inter", 4]))
    if "k1-inter" in selected:
        for value in parse_int_list(args.k1_inter_values):
            items.append(("k1-inter", str(value), ["--k1-intra", 5, "--k1-inter", value, "--k2-intra", 2, "--k2-inter", 4]))
    if "k2" in selected:
        for left, right in parse_k2_pairs(args.k2_pairs):
            items.append(("k2", f"{left}-{right}", ["--k1-intra", 5, "--k1-inter", 20, "--k2-intra", left, "--k2-inter", right]))
    return items


def record(args, scene, sweep_name, label, cmd):
    append_csv(args.command_csv, {
        "experiment": "fig4",
        "scene": scene,
        "dataset": args.dataset,
        "sweep": sweep_name,
        "value": label,
        "command": " ".join(map(str, cmd)),
    })


def record_result(args, scene, sweep_name, label, log_path):
    metrics = extract_reid_metrics(log_path)
    if not metrics:
        return
    row = {
        "experiment": "fig4",
        "scene": scene,
        "dataset": args.dataset,
        "sweep": sweep_name,
        "value": label,
    }
    row.update(metrics)
    append_csv(args.results_csv, row)


def main():
    args = parse_args()
    ensure_dataset(args.dataset)
    scenes = ["clustering", "reranking"] if args.scene == "both" else [args.scene]

    for scene in scenes:
        if scene == "reranking" and not args.bot_checkpoint:
            raise ValueError("--bot-checkpoint is required for re-ranking scene")
        for sweep_name, label, extra_flags in sweep_items(args):
            if scene == "clustering":
                cmd, logs_dir = clustering_command(args, sweep_name, label, extra_flags)
                log_path = logs_dir / "log.txt"
            else:
                cmd, logs_dir = rerank_command(args, sweep_name, label, extra_flags)
                log_path = logs_dir / "log_test.txt"
            record(args, scene, sweep_name, label, cmd)
            run_command(cmd, dry_run=args.dry_run, gpu=args.gpu)
            if not args.dry_run:
                record_result(args, scene, sweep_name, label, log_path)


if __name__ == "__main__":
    main()
