from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.decision_tree import SimpleDecisionTree, TopKDecisionTree
from src.metrics import (
    calculate_accuracy,
    calculate_f1_score,
    calculate_precision,
    calculate_recall,
    calculate_tree_depth,
    count_leaves,
    count_nodes,
)
from src.utils import binarize_features

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "experiments")


class SklearnBaselineWrapper:
    """Wrapper around scikit-learn's DecisionTreeClassifier for baseline comparison."""

    def __init__(self, max_depth: int = 4, random_state: int = 42):
        self.clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)


def make_toy_dataset():
    X = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1], dtype=int)
    return X, y


def load_iris_raw_dataset():
    iris = load_iris()
    mask = iris.target < 2  # Binary subset: Setosa (0) vs Versicolor (1)
    X = iris.data[mask]
    y = iris.target[mask]
    return X, y.astype(int)


def load_breast_cancer_raw_dataset():
    dataset = load_breast_cancer()
    return dataset.data, dataset.target.astype(int)


def build_raw_datasets():
    return {
        "toy": make_toy_dataset(),
        "iris_binary": load_iris_raw_dataset(),
        "breast_cancer_binary": load_breast_cancer_raw_dataset(),
    }


def train_and_evaluate(model_factory, X_train, X_test, y_train, y_test, n_repetitions: int = 3):
    train_times = []
    pred_times = []
    fitted_model = None
    first_preds = None

    for rep in range(n_repetitions):
        model = model_factory()

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        t1 = time.perf_counter()
        train_times.append(t1 - t0)

        t2 = time.perf_counter()
        preds = model.predict(X_test)
        t3 = time.perf_counter()
        pred_times.append(t3 - t2)

        if rep == 0:
            fitted_model = model
            first_preds = preds

    accuracy = calculate_accuracy(y_test, first_preds)
    precision = calculate_precision(y_test, first_preds)
    recall = calculate_recall(y_test, first_preds)
    f1 = calculate_f1_score(y_test, first_preds)

    if hasattr(fitted_model, "root"):
        depth = calculate_tree_depth(fitted_model.root)
        nodes = count_nodes(fitted_model.root)
        leaves = count_leaves(fitted_model.root)
    elif hasattr(fitted_model, "clf"):
        depth = fitted_model.clf.get_depth()
        nodes = fitted_model.clf.tree_.node_count
        leaves = fitted_model.clf.get_n_leaves()
    else:
        depth = 0
        nodes = 0
        leaves = 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_time_mean": float(np.mean(train_times)),
        "training_time_std": float(np.std(train_times)),
        "prediction_time_mean": float(np.mean(pred_times)),
        "prediction_time_std": float(np.std(pred_times)),
        "tree_depth": depth,
        "number_of_nodes": nodes,
        "number_of_leaves": leaves,
    }


def run_experiments(max_depth: int = 4, test_size: float = 0.3, random_state: int = 42, n_repetitions: int = 3):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    rows = []
    raw_datasets = build_raw_datasets()

    for dataset_name, (X_raw, y_raw) in raw_datasets.items():
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw,
            y_raw,
            test_size=test_size,
            random_state=random_state,
            stratify=y_raw if len(np.unique(y_raw)) > 1 else None,
        )

        # STRICT NO DATA LEAKAGE: fit threshold ONLY on training set
        X_train, thresholds = binarize_features(X_train_raw)
        X_test, _ = binarize_features(X_test_raw, thresholds=thresholds)

        models = [
            ("Greedy", lambda: SimpleDecisionTree(max_depth=max_depth), 1),
            ("TopK", lambda: TopKDecisionTree(k=2, max_depth=max_depth), 2),
            ("TopK", lambda: TopKDecisionTree(k=3, max_depth=max_depth), 3),
            ("SklearnDecisionTree", lambda: SklearnBaselineWrapper(max_depth=max_depth, random_state=random_state), "NA"),
        ]

        for model_name, factory, k in models:
            metrics = train_and_evaluate(factory, X_train, X_test, y_train, y_test, n_repetitions=n_repetitions)
            row = {
                "dataset": dataset_name,
                "model": model_name,
                "k": k,
                **metrics,
            }
            rows.append(row)

    columns = [
        "dataset",
        "model",
        "k",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "training_time_mean",
        "training_time_std",
        "prediction_time_mean",
        "prediction_time_std",
        "tree_depth",
        "number_of_nodes",
        "number_of_leaves",
    ]
    results = pd.DataFrame(rows)[columns]

    results_path = os.path.join(EXPERIMENTS_DIR, "results.csv")
    benchmark_path = os.path.join(RESULTS_DIR, "benchmark_scores.csv")
    results.to_csv(results_path, index=False)
    results.to_csv(benchmark_path, index=False)

    generate_plots(results)
    return results


def generate_plots(results: pd.DataFrame):
    plt.style.use("ggplot")
    datasets = results["dataset"].unique()
    models_labels = ["Greedy (k=1)", "TopK (k=2)", "TopK (k=3)", "Sklearn (k=NA)"]
    model_keys = [("Greedy", 1), ("TopK", 2), ("TopK", 3), ("SklearnDecisionTree", "NA")]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # 1. Accuracy Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(datasets))
    width = 0.18

    for idx, (m_name, k_val) in enumerate(model_keys):
        sub = results[(results["model"] == m_name) & (results["k"] == k_val)]
        accs = [sub[sub["dataset"] == d]["accuracy"].values[0] if d in sub["dataset"].values else 0 for d in datasets]
        rects = ax.bar(x + idx * width - width * 1.5, accs, width, label=models_labels[idx], color=colors[idx])
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Accuracy")
    ax.set_title("Graph 1 — Accuracy Comparison Across Datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.15)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_comparison.png"), dpi=200)
    plt.close()

    # 2. Precision, Recall, and F1-score
    metrics = ["precision", "recall", "f1_score"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for idx, (m_name, k_val) in enumerate(model_keys):
            sub = results[(results["model"] == m_name) & (results["k"] == k_val)]
            vals = [sub[sub["dataset"] == d][metric].values[0] if d in sub["dataset"].values else 0 for d in datasets]
            ax.bar(x + idx * width - width * 1.5, vals, width, label=models_labels[idx], color=colors[idx])

        ax.set_title(f"{metric.replace('_', ' ').title()}")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim(0, 1.15)

    axes[0].set_ylabel("Score")
    axes[0].legend(fontsize=8)
    fig.suptitle("Graph 2 — Classification Metrics (Precision, Recall, F1)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "classification_metrics_comparison.png"), dpi=200)
    plt.close()

    # 3. Training Time Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, (m_name, k_val) in enumerate(model_keys):
        sub = results[(results["model"] == m_name) & (results["k"] == k_val)]
        times = [sub[sub["dataset"] == d]["training_time_mean"].values[0] if d in sub["dataset"].values else 0 for d in datasets]
        errs = [sub[sub["dataset"] == d]["training_time_std"].values[0] if d in sub["dataset"].values else 0 for d in datasets]
        ax.bar(x + idx * width - width * 1.5, times, width, yerr=errs, capsize=3, label=models_labels[idx], color=colors[idx])

    ax.set_ylabel("Training Time (seconds)")
    ax.set_title("Graph 3 — Mean Training Time Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_time_comparison.png"), dpi=200)
    plt.close()

    # 4. Prediction Time Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, (m_name, k_val) in enumerate(model_keys):
        sub = results[(results["model"] == m_name) & (results["k"] == k_val)]
        times = [sub[sub["dataset"] == d]["prediction_time_mean"].values[0] if d in sub["dataset"].values else 0 for d in datasets]
        errs = [sub[sub["dataset"] == d]["prediction_time_std"].values[0] if d in sub["dataset"].values else 0 for d in datasets]
        ax.bar(x + idx * width - width * 1.5, times, width, yerr=errs, capsize=3, label=models_labels[idx], color=colors[idx])

    ax.set_ylabel("Prediction Time (seconds)")
    ax.set_title("Graph 4 — Mean Prediction Time Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "prediction_time_comparison.png"), dpi=200)
    plt.close()

    # 5. Tree Depth Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, (m_name, k_val) in enumerate(model_keys):
        sub = results[(results["model"] == m_name) & (results["k"] == k_val)]
        depths = [sub[sub["dataset"] == d]["tree_depth"].values[0] if d in sub["dataset"].values else 0 for d in datasets]
        ax.bar(x + idx * width - width * 1.5, depths, width, label=models_labels[idx], color=colors[idx])

    ax.set_ylabel("Tree Depth")
    ax.set_title("Graph 5 — Tree Depth Comparison Across Datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "tree_depth_comparison.png"), dpi=200)
    plt.close()

    # 6. Node Count Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, (m_name, k_val) in enumerate(model_keys):
        sub = results[(results["model"] == m_name) & (results["k"] == k_val)]
        nodes = [sub[sub["dataset"] == d]["number_of_nodes"].values[0] if d in sub["dataset"].values else 0 for d in datasets]
        ax.bar(x + idx * width - width * 1.5, nodes, width, label=models_labels[idx], color=colors[idx])

    ax.set_ylabel("Total Number of Nodes")
    ax.set_title("Graph 6 — Node Count Comparison Across Datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "node_count_comparison.png"), dpi=200)
    plt.close()

    # 7. Accuracy vs Training Time Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for idx, row in results.iterrows():
        label = f"{row['model']} (k={row['k']}) [{row['dataset']}]"
        ax.scatter(row["training_time_mean"], row["accuracy"], s=100, alpha=0.8)
        ax.annotate(label, xy=(row["training_time_mean"], row["accuracy"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)

    ax.set_xlabel("Training Time (seconds)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Graph 7 — Accuracy vs Training Time Trade-off")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_vs_training_time.png"), dpi=200)
    plt.close()

    # 8. Effect of k (Top-k Hyperparameter Variation)
    ds_target = "breast_cancer_binary" if "breast_cancer_binary" in datasets else datasets[0]
    sub_k = results[(results["dataset"] == ds_target) & (results["model"].isin(["Greedy", "TopK"]))]
    sub_k = sub_k.sort_values("k")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    k_vals = sub_k["k"].astype(int).values
    acc_vals = sub_k["accuracy"].values
    time_vals = sub_k["training_time_mean"].values

    color_acc = "#1f77b4"
    ax1.set_xlabel("k (Number of Top Features Choice)")
    ax1.set_ylabel("Accuracy", color=color_acc)
    ax1.plot(k_vals, acc_vals, marker="o", color=color_acc, linewidth=2, label="Accuracy")
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_xticks(k_vals)

    ax2 = ax1.twinx()
    color_time = "#d62728"
    ax2.set_ylabel("Training Time (s)", color=color_time)
    ax2.plot(k_vals, time_vals, marker="s", linestyle="--", color=color_time, linewidth=2, label="Training Time")
    ax2.tick_params(axis="y", labelcolor=color_time)

    plt.title(f"Graph 8 — Effect of k on Accuracy & Training Time ({ds_target})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "top_k_effect.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    results = run_experiments()
    print("\nExperiment Execution Complete.")