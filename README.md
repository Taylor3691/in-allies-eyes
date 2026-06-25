# in-allies-eyes

This repository contains a working CA-Jaccard person re-identification setup integrated with a Bag-of-Tricks (BoT) training and testing pipeline. The codebase has been adapted to support running experiments on **Market1501** and the **CUHK03** dataset.

All commands assume they are run from the repository root with the project environment active.

## Prerequisites

### Environment Setup

Create a Python 3.11+ environment, activate it, and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For faster package resolution and installation, you can use `uv`:

```bash
pip install uv
uv pip install -r requirements.txt
```

### Dataset Downloads

This codebase supports downloading and extracting both Market1501 and CUHK03 datasets:

```bash
# Download and extract Market1501
python scripts/download_datasets.py market1501

# Download and extract CUHK03
python scripts/download_datasets.py cuhk03
```

By default, archives are stored in `data/_downloads` and extracted/mapped to:
- `data/market1501/Market-1501-v15.09.15`
- `data/cuhk03`

On Kaggle platforms, the downloader automatically detects `/kaggle/input` datasets and symlinks the parent of `cuhk03_release` to `data/cuhk03` directly.

Downloaded archives are removed after successful extraction. Use `--keep-archives` if you want to keep the zip file:

```bash
python scripts/download_datasets.py market1501 --keep-archives
```

### Pretrained Weights

Download the pretrained models:

```bash
# Download demo checkpoints for Market1501
python scripts/download_pretrained_models.py demo

# Download ImageNet-pretrained ResNet50 backbone (required for training)
python scripts/download_pretrained_models.py resnet50
```

Pretrained backbones and checkpoints are saved under the `pretrained_models/` directory.

---

## Demo App

[Placeholder for Demo App description and instructions]

---

## Experiments

The following sections are for reproducing the experiments and analyses:

- **Table 3**: CKRNNs/CLQE/CAJ ablations
- **Figure 3**: Neighbor analysis over clustering epochs
- **Figure 4**: Parameter analysis

### Training BoT on CUHK03

A dedicated CUHK03 configuration file is provided at `thirdparty/bot/configs/softmax_triplet_with_center_cuhk03.yml`. It uses standard settings optimized for CUHK03.

To train the Bag-of-Tricks model on CUHK03:

```bash
cd thirdparty/bot
python tools/train.py \
  --config_file configs/softmax_triplet_with_center_cuhk03.yml
```

The resulting checkpoint can be evaluated using the CAJ testing pipeline by passing `--checkpoint-format bot`.

---

### Table 3 Ablation

Run the clustering ablation for Market1501:

```bash
python scripts/run_tab3_ablation.py \
  --dataset market1501 \
  --scene clustering
```

Run the clustering ablation for CUHK03:

```bash
python scripts/run_tab3_ablation.py \
  --dataset cuhk03 \
  --scene clustering
```

To skip clustering training and evaluate existing CAJ checkpoints:

```bash
python scripts/run_tab3_ablation.py \
  --dataset market1501 \
  --scene clustering \
  --cluster-checkpoint "logs/experiments/tab3/clustering/{dataset}/{variant}/model_best.pth.tar"
```

BoT re-ranking ablation needs a BoT checkpoint:

```bash
python scripts/run_tab3_ablation.py \
  --dataset market1501 \
  --scene reranking \
  --bot-checkpoint pretrained_models/market_resnet50_model_120_rank1_945.pth
```

Use `--dry-run` first to inspect the generated command list. Commands are logged to `results/tab3_commands.csv` and results are parsed to `results/tab3_results.csv`.

---

### Figure 3 Neighbor Analysis

Neighbor analysis has been merged into the Table 3 Clustering Ablation script (`run_tab3_ablation.py`) to avoid duplicate training runs. When you run the Table 3 clustering ablation, the neighbor-analysis files are automatically generated alongside training:

```bash
# Market1501
python scripts/run_tab3_ablation.py --dataset market1501 --scene clustering

# CUHK03
python scripts/run_tab3_ablation.py --dataset cuhk03 --scene clustering
```

---

### Figure 4 Parameter Analysis

Sweep hyperparameters (`k1-intra`, `k1-inter`, and `k2` variations) to analyze sensitivity:

```bash
python scripts/run_fig4_params.py \
  --dataset cuhk03 \
  --scene clustering \
  --sweep all
```

---

### Plotting

Plot Figure 3-style curves from neighbor-analysis CSV files:

```bash
python scripts/plot_experiments.py \
  --kind fig3 \
  --input logs/experiments/tab3/clustering/market1501/*/neighbor_analysis.csv
```

Plot Figure 4-style curves from sweep results:

```bash
python scripts/plot_experiments.py \
  --kind fig4 \
  --input results/fig4_results.csv \
  --metric mAP
```

Render a Table 3-style Markdown table:

```bash
python scripts/plot_experiments.py \
  --kind tab3 \
  --input results/tab3_results.csv
```

All figures and tables are saved under `results/figures/` by default.

---

## Acknowledgement

This repository builds on the official [CA-Jaccard](https://github.com/chenyiyuu/CA-Jaccard) implementation and includes code adapted from Cluster Contrast and [BoT](https://github.com/michuanhaohao/reid-strong-baseline) for person re-identification experiments.
