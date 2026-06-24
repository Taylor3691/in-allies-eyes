#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path


FIG3_METRICS = [
    ("avg_inter_camera_proportion", "Proportion"),
    ("avg_inter_camera_weight", "Weight"),
    ("avg_same_id_accuracy", "Accuracy"),
]

SCENE_LABELS = {
    "clustering": "Clustering",
    "reranking": "Reranking",
}

MODE_LABELS = {
    "baseline": "Baseline",
    "ckrnns": "CKRNNs",
    "clqe": "CLQE",
    "caj": "CA-Jaccard",
}

VARIANT_ORDER = {
    "baseline": 0,
    "ckrnns": 1,
    "clqe": 2,
    "caj": 3,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CA-Jaccard experiment CSV outputs")
    parser.add_argument("--kind", choices=["fig3", "fig4", "tab3"], required=True)
    parser.add_argument("--input", nargs="+", required=True, help="CSV file(s)")
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--metric", default="mAP", help="metric column for fig4 CSVs")
    return parser.parse_args()


def read_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def require_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it in the active environment.") from exc
    return plt


def as_float(row, key):
    value = row.get(key, "")
    return float(value) if value != "" else 0.0


def plot_fig3(rows, output_dir):
    plt = require_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("dataset", ""), row.get("mode", ""))].append(row)

    datasets = sorted({key[0] for key in grouped.keys()})
    for dataset in datasets:
        dataset_rows = {mode: values for (ds, mode), values in grouped.items() if ds == dataset}
        for metric, ylabel in FIG3_METRICS:
            plt.figure(figsize=(4, 3))
            for mode, values in sorted(
                dataset_rows.items(),
                key=lambda item: (VARIANT_ORDER.get(item[0], 99), item[0]),
            ):
                values = sorted(values, key=lambda r: int(r["epoch"]))
                epochs = [int(row["epoch"]) for row in values]
                ys = [as_float(row, metric) for row in values]
                plt.plot(epochs, ys, label=MODE_LABELS.get(mode, mode), linewidth=1)
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout(pad=0.7)
            out = output_dir / f"fig3_{dataset}_{metric}.png"
            plt.savefig(out, dpi=400)
            plt.close()
            print(f"saved: {out}")


def plot_fig4(rows, output_dir, metric):
    plt = require_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("dataset", ""), row.get("sweep", ""))].append(row)

    def sort_value(row):
        value = row.get("value", "")
        if "-" in value:
            return tuple(int(part) for part in value.split("-", 1))
        try:
            return (int(value),)
        except ValueError:
            return (value,)

    for (dataset, sweep), values in sorted(grouped.items()):
        plt.figure(figsize=(4, 3))
        scene_rows = defaultdict(list)
        for row in values:
            scene_rows[row.get("scene", "")].append(row)
        for scene, scene_values in sorted(scene_rows.items()):
            scene_values = sorted(scene_values, key=sort_value)
            xs = [row.get("value", "").replace('-', '/') for row in scene_values]
            ys = [as_float(row, metric) for row in scene_values]
            plt.plot(xs, ys, label=SCENE_LABELS.get(scene, scene), marker="o", markersize=4, linewidth=1)
            for x, y in zip(xs, ys):
                plt.annotate(
                    f"{y:.1f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 5 if scene == 'clustering' else -12),
                    ha="center",
                    fontsize=8,
                )
        plt.ylabel(f"{metric}{' (%)' if metric == 'mAP' else ''}")
        plt.legend()
        plt.tight_layout(pad=0.7)
        plt.grid(True)
        out = output_dir / f"fig4_{dataset}_{sweep}.png"
        plt.savefig(out, dpi=400)
        plt.close()
        print(f"saved: {out}")


def render_tab3(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_order = {"clustering": 0, "reranking": 1}
    rows = sorted(
        rows,
        key=lambda r: (
            scene_order.get(r.get("scene", ""), 99),
            r.get("scene", ""),
            r.get("dataset", ""),
            VARIANT_ORDER.get(r.get("variant", ""), 99),
            r.get("variant", ""),
        ),
    )
    metric_headers = [
        key
        for key in ("mAP", "Rank-1", "Rank-5")
        if any(row.get(key, "") != "" for row in rows)
    ]
    headers = ["Dataset", "Variant"] + metric_headers
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    current_scene = None
    for row in rows:
        scene = row.get("scene", "")
        if scene != current_scene:
            current_scene = scene
            lines.append(f"| {SCENE_LABELS.get(scene, scene)} |")
        values = [
            row.get("dataset", ""),
            row.get("variant", ""),
            *[row.get(header, "") for header in metric_headers],
        ]
        lines.append("| " + " | ".join(values) + " |")
    out = output_dir / "tab3_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"saved: {out}")


def main():
    args = parse_args()
    rows = read_rows(args.input)
    output_dir = Path(args.output_dir)
    if args.kind == "fig3":
        plot_fig3(rows, output_dir)
    elif args.kind == "fig4":
        plot_fig4(rows, output_dir, args.metric)
    else:
        render_tab3(rows, output_dir)


if __name__ == "__main__":
    main()
