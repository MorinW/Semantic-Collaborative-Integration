# Rethinking Semantic-Collaborative Integration

This repository contains the anonymous official implementation for the paper: **"Rethinking Semantic-Collaborative Integration: Why Alignment Is Not Enough"**.

We provide a complete, reproducible pipeline including data splitting, model training, and specific **diagnostic probes** to validate the claims made in our perspective.

## 🗺️ Roadmap to Claims

As a Perspective paper, our code is organized to empirically validate three core claims regarding the relationship between Semantic (LLM) and Collaborative (ID) spaces:

| Claim in Paper | Description | Corresponding Code / Analysis |
| --- | --- | --- |
| **Claim 1** | **Complementarity:** Semantic and Collaborative views capture orthogonal signals (low overlap). | `evaluate_comp.py` (Jaccard, Unique Hits) |
| **Claim 2** | **Limits of Alignment:** Low-capacity mapping cannot recover collaborative geometry (Fitting vs. Generalization). | `analyze_alignment.py`<br>`analyze_generalization.py` |
| **Claim 3** | **Paradigm Shift:** Naive Fusion (Union) outperforms complex Homomorphism/Alignment paradigms. | `run_naive_fusion.py` vs. `run_alpharec.py` & `run_baseline.py` |

---

## 📂 Repository Structure

```text
.
├── dataset/                    # Pre-processed 5-core data & Embeddings
│   ├── movies/                 # Contains: movies.inter, movies_item_embeddings.npy, item_map.json
│   ├── games/
│   └── amazon-book/
├── Case study/                 # [Case Study Programs]
│   ├── case1.py                # Item Align Analysis: KNN comparison for Hot/Tail anchors
│   ├── case2.py                # Global Dist: t-SNE visualization with stratified sampling
│   ├── case3.py                # User View: Dual-view visualization (Behavior vs Content)
│   └── case4&5.py                # Cold Start: Mining cases where Semantic hits & CF misses
├── saved/                      # Saved model checkpoints (.pth)
├── stats/                      # Statistical logs and Popularity JSONs
├── split_dataset.py            # [Data Prep] Ensures strict train/test consistency
├── run_baseline.py             # [Baselines] LightGCN, NCL (RecBole-based)
├── run_simgcl.py               # [Claim 3] Contrastive Learning Baseline
├── run_alpharec.py             # [Claim 3] Homomorphism Paradigm Baseline
├── run_semantic.py             # [Training] Semantic Baseline / Encoder
├── run_naive_fusion.py         # [Claim 3] Our Model: Naive Fusion
├── group_items.py              # [Analysis] Generate Popularity Groups
├── evaluate_naive_fusion.py    # [Analysis] Stratified Eval & Oracle Bound
├── evaluate_comp.py            # [Claim 1] Complementarity Diagnostics
├── analyze_alignment.py        # [Claim 2] Probe 1: Capacity & Fitting Analysis
├── analyze_generalization.py   # [Claim 2] Probe 2: Inductive Generalization Analysis
└── requirements.txt            # Dependencies

```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
pip install -r requirements.txt

```

### 2. Data Preparation

To ensure the reproducibility of our experimental results, strict data consistency is required.

#### A. Data Source & Pre-processing

We utilize the **[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)** dataset (Movies, Books, Games) with standard **5-core** filtering. Textual representations (1024-dim) are encoded using **BGE-M3**.

> **✅ For your convenience:** The pre-processed interaction files (`.inter`) and item embeddings (`.npy`) are **already provided** in the `dataset/{DATASET}/` directory.

#### B. Unified Data Splitting

Run the splitting script to generate `train/valid/test` sets with a fixed seed.

```bash
# Example for Movies
python split_dataset.py --dataset movies --seed 2020

```

---

## 🧪 Stage 1: Training & Embedding Export

All training scripts will **automatically export** the learned User/Item embeddings to `dataset/{DATASET}/` upon completion.

### A. Train Collaborative Baselines (LightGCN)

We use RecBole to train ID-based collaborative filtering models.

```bash
python run_baseline.py --dataset movies --model LightGCN

```

* **Output:** `dataset/movies/movies_user_embeddings-lightgcn.npy`

### B. Train Homomorphism Baseline (AlphaRec)

Reproducing the **Homomorphism Paradigm** (as discussed in Claim 3).
*Note: AlphaRec assumes a structural homomorphism where collaborative signals can be recovered from semantic embeddings via projection. It does **not** employ explicit alignment losses.*

```bash
python run_alpharec.py --dataset movies --hidden_size 64 --train_use_gcn

```

* **Output:** `dataset/movies/movies_user_embeddings-AlphaRec.npy`

### C. Train Semantic Baseline

A pure content-based model trained with InfoNCE. This also serves as the source embedding for alignment probes.

```bash
python run_semantic.py --dataset movies --proj_dim 64

```

* **Output:** `dataset/movies/movies_user_embeddings-proj64.npy`

### D. Train Our Model (Naive Fusion)

Reproducing the main results (**Table 6**) using the "Norm-Concat-Norm" strategy.

```bash
python run_naive_fusion.py \
  --dataset movies \
  --hard_neg_factor 2 \
  --temperature 0.15 \
  --epochs 200

```

* **Output:** `dataset/movies/movies_user_embeddings-SimpleHard.npy`

---

## 🔍 Stage 2: Analytical Probes

### Analysis 1: The Alignment Probe (Claim 2)

To investigate the **"Global Low-Complexity Alignment Hypothesis"**, we use two probes to test if semantic embeddings can be mapped to collaborative geometry.

**Probe A: Capacity & Fitting Analysis**
Tests if a mapping *can* theoretically be learned on the full item set. High  here indicates the model has enough capacity to fit the data.

```bash
python analyze_alignment.py \
  --dataset movies \
  --sem_suffix proj64 \
  --col_suffix pco64

```

* **Key Metrics:**  (Reconstruction), GeoJac (Neighborhood Overlap).

**Probe B: Inductive Generalization Analysis**
Tests if the learned mapping *generalizes* to unseen items. A large drop in performance compared to Probe A indicates structural mismatch (overfitting to seen items).

```bash
python analyze_generalization.py \
  --dataset movies \
  --sem_suffix proj64 \
  --col_suffix pco64 \
  --train_ratio 0.8

```

* **Key Metrics:** Drop% (Performance loss on unseen items vs. Oracle CF).

### Analysis 2: Popularity-Stratified Analysis (Table 8)

To reproduce the breakdown of performance across Cold, Mid, and Hot groups.

```bash
# 1. Generate Groups
python group_items.py --dataset movies

# 2. Run Evaluation
python evaluate_naive_fusion.py --dataset movies --emb_suffix SimpleHard

```

### Analysis 3: Complementarity Diagnostics (Claim 1)

To verify that the Semantic View and Collaborative View retrieve *different* sets of items.

```bash
python evaluate_comp.py \
  --dataset movies \
  --suffix_a lightgcn \
  --suffix_b SimpleHard

```

