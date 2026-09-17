# Harnessing the Power of Choices in Decision Tree Learning — Refined Implementation

A lightweight educational Python implementation and fair benchmark suite inspired by the NeurIPS paper:

> **"Harnessing the Power of Choices in Decision Tree Learning"**

---

## 📌 Project Overview

Standard decision tree algorithms (e.g., ID3, C4.5, CART) are **greedy**: at every node, they evaluate all available features and pick the single best split based on Information Gain or Gini Impurity. While fast, this greedy choice can be myopic—a locally top-ranked split is not guaranteed to yield the globally optimal decision tree.

This repository implements a **Top-k Decision Tree** variant alongside a classic **Greedy Decision Tree**:
- **Greedy (`k=1`)**: Always picks the feature with the highest Information Gain.
- **Top-k (`k=2, k=3`)**: Identifies the top $k$ feature candidates ranked by Information Gain, builds candidate subtrees, and selects the split that achieves the best local performance.

---

## 🏗️ Project Structure

```text
decision-tree-top-k/
├── src/
│   ├── decision_tree.py     # SimpleDecisionTree (Greedy) & TopKDecisionTree (k=1,2,3)
│   ├── metrics.py           # Accuracy, Precision, Recall, F1, Depth, Node Count, Leaf Count
│   └── utils.py             # Feature binarization (strict train-only thresholding)
├── experiments/
│   ├── run_experiments.py   # Benchmark runner, dataset loaders, timing, plotting
│   └── results.csv          # Raw experiment results CSV
├── results/
│   ├── benchmark_scores.csv # Detailed benchmark metrics
│   ├── accuracy_comparison.png
│   ├── classification_metrics_comparison.png
│   ├── training_time_comparison.png
│   ├── prediction_time_comparison.png
│   ├── tree_depth_comparison.png
│   ├── node_count_comparison.png
│   ├── accuracy_vs_training_time.png
│   └── top_k_effect.png
├── docs/
│   ├── refinement_plan.md   # Project refinement and audit plan
│   ├── benchmark_analysis.md# Detailed experimental analysis report
│   ├── paper_summary.md     # Summary of the research paper
│   └── implementation_notes.md
├── tests/
│   └── test_models.py       # 15 unit tests (Pytest)
├── main.py                  # Entry point: runs experiments, prints table, outputs graphs
├── requirements.txt         # Project dependencies
└── README.md                # Documentation & Viva guide
```

---

## ⚡ Quick Start

### 1. Installation

Ensure Python 3.9+ is installed, then install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Running Benchmarks & Generating Outputs

Run the complete experiment pipeline:

```bash
python main.py
```

This command will:
1. Run sanity checks on the toy binary dataset.
2. Train all models across 3 datasets (`toy`, `iris_binary`, `breast_cancer_binary`).
3. Print a 12-column ASCII benchmark table in the terminal.
4. Save CSV results to `results/benchmark_scores.csv` and `experiments/results.csv`.
5. Generate and save 8 PNG graphs in `results/`.

### 3. Running Unit Tests

Execute the Pytest suite:

```bash
python -m pytest
```

---

## 📊 Benchmark Results

Evaluated with `RANDOM_STATE = 42`, `test_size = 0.3`, `max_depth = 4`, and 3 timing repetitions:

| Dataset | Model | k | Accuracy | Precision | Recall | F1-Score | Train Time (s) | Pred Time (s) | Depth | Nodes | Leaves |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `toy` | Greedy | 1 | 0.7500 | 0.8333 | 0.7500 | 0.7333 | 0.000541 | 0.000006 | 3 | 5 | 3 |
| `toy` | TopK | 2 | 0.7500 | 0.8333 | 0.7500 | 0.7333 | 0.007450 | 0.000007 | 3 | 5 | 3 |
| `toy` | TopK | 3 | 0.7500 | 0.8333 | 0.7500 | 0.7333 | 0.007534 | 0.000007 | 3 | 5 | 3 |
| `toy` | Sklearn | NA | 0.7500 | 0.8333 | 0.7500 | 0.7333 | 0.001665 | 0.000192 | 2 | 5 | 3 |
| `iris_binary` | Greedy | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000791 | 0.000018 | 2 | 3 | 2 |
| `iris_binary` | TopK | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.001686 | 0.000015 | 2 | 3 | 2 |
| `iris_binary` | TopK | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.008025 | 0.000015 | 2 | 3 | 2 |
| `iris_binary` | Sklearn | NA | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000602 | 0.000104 | 1 | 3 | 2 |
| `breast_cancer_binary` | Greedy | 1 | 0.8713 | 0.8778 | 0.8713 | 0.8726 | 0.024582 | 0.000150 | 4 | 15 | 8 |
| `breast_cancer_binary` | TopK | 2 | 0.8889 | 0.9023 | 0.8889 | 0.8904 | 0.194733 | 0.000160 | 4 | 15 | 8 |
| `breast_cancer_binary` | TopK | 3 | **0.9123** | **0.9193** | **0.9123** | **0.9132** | 0.405740 | 0.000143 | 4 | 15 | 8 |
| `breast_cancer_binary` | Sklearn | NA | **0.9123** | **0.9193** | **0.9123** | **0.9132** | 0.001136 | 0.000152 | 4 | 29 | 15 |

---

## 📈 Generated Visualizations

All graphs are saved automatically in `results/`:

1. `accuracy_comparison.png`: Accuracy comparison across models & datasets.
2. `classification_metrics_comparison.png`: Grouped Precision, Recall, and F1-score chart.
3. `training_time_comparison.png`: Mean training runtime with error bars.
4. `prediction_time_comparison.png`: Inference latency comparison.
5. `tree_depth_comparison.png`: Maximum tree depth per model.
6. `node_count_comparison.png`: Total tree node count per model.
7. `accuracy_vs_training_time.png`: Scatter plot showing accuracy vs runtime trade-off.
8. `top_k_effect.png`: Dual-axis plot showing accuracy boost and runtime cost as $k$ increases ($k=1 \to 3$).

---

## 🔒 Data Leakage Prevention Rule

To guarantee strict benchmark fairness, median thresholds for binary feature transformation are **calculated exclusively on training data** (`X_train`) after dataset splitting. These exact thresholds are then applied to transform both `X_train` and `X_test`.

---

## 🎓 College Presentation & Viva Guide

### Q1: What problem does Top-k Decision Tree solve?
**Answer**: Classic decision trees use greedy heuristics to pick the single best split feature at each node. Greedy choices can lead to sub-optimal subtrees. Top-k decision trees expand search locally by considering the top $k$ candidate features before choosing a split, discovering trees with better overall generalization.

### Q2: Why is `k=1` equivalent to Greedy?
**Answer**: When $k=1$, the algorithm evaluates only 1 candidate feature—the one with the highest Information Gain. This is identical to classic greedy decision tree split selection.

### Q3: How are candidate splits evaluated in this Top-k implementation?
**Answer**: For each of the top $k$ features, a candidate subtree is constructed recursively. The candidate split yielding the highest local classification accuracy on the node's training subset is selected, using Information Gain as a tie-breaker.

### Q4: What is the trade-off of using larger $k$?
**Answer**: Higher $k$ values increase training runtime because $k$ candidate subtrees are evaluated at each non-leaf node. However, prediction runtime remains identical during inference.

---

## ⚠️ Educational Scope & Limitations

This project is a **lightweight educational approximation** designed for clarity and viva presentation. It does not attempt to reproduce the full theoretical proofs or C++ solver optimizations from the original NeurIPS paper.
