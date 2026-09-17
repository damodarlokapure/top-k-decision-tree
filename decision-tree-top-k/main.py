from __future__ import annotations

import os

import pandas as pd

from experiments.run_experiments import make_toy_dataset, run_experiments
from src.decision_tree import SimpleDecisionTree, TopKDecisionTree
from src.metrics import calculate_accuracy, calculate_tree_depth, count_nodes


def train_and_report_toy(model, X, y, label):
    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = calculate_accuracy(y, predictions)
    depth = calculate_tree_depth(model.root)
    nodes = count_nodes(model.root)
    print(f"  {label:<18}: accuracy={accuracy:.3f}, depth={depth}, nodes={nodes}")


def print_results_table(results: pd.DataFrame):
    columns = [
        "dataset",
        "model",
        "k",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "training_time_mean",
        "prediction_time_mean",
        "tree_depth",
        "number_of_nodes",
        "number_of_leaves",
    ]

    print("\n" + "=" * 125)
    print(f"{'BENCHMARK EXPERIMENT RESULTS':^125}")
    print("=" * 125)
    header = (
        f"{'Dataset':<22} {'Model':<20} {'k':<4} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} "
        f"{'F1-Score':<10} {'Train Time(s)':<14} {'Pred Time(s)':<14} {'Depth':<6} {'Nodes':<6} {'Leaves':<6}"
    )
    print(header)
    print("-" * 125)

    for _, row in results[columns].iterrows():
        k_str = str(row['k'])
        train_time_str = f"{row['training_time_mean']:.6f}"
        pred_time_str = f"{row['prediction_time_mean']:.6f}"
        print(
            f"{str(row['dataset']):<22} "
            f"{str(row['model']):<20} "
            f"{k_str:<4} "
            f"{row['accuracy']:<10.4f} "
            f"{row['precision']:<10.4f} "
            f"{row['recall']:<10.4f} "
            f"{row['f1_score']:<10.4f} "
            f"{train_time_str:<14} "
            f"{pred_time_str:<14} "
            f"{int(row['tree_depth']):<6} "
            f"{int(row['number_of_nodes']):<6} "
            f"{int(row['number_of_leaves']):<6}"
        )
    print("=" * 125)


def print_summary_statistics(results: pd.DataFrame):
    print("\n--- Summary Statistics & Findings ---")

    datasets = results["dataset"].unique()
    for ds in datasets:
        ds_results = results[results["dataset"] == ds]
        best_acc_row = ds_results.sort_values("accuracy", ascending=False).iloc[0]
        fastest_row = ds_results.sort_values("training_time_mean", ascending=True).iloc[0]

        greedy_row = ds_results[ds_results["model"] == "Greedy"].iloc[0]
        topk2_row = ds_results[(ds_results["model"] == "TopK") & (ds_results["k"] == 2)]
        topk3_row = ds_results[(ds_results["model"] == "TopK") & (ds_results["k"] == 3)]

        print(f"\nDataset: [{ds}]")
        print(f"  • Best Accuracy : {best_acc_row['model']} (k={best_acc_row['k']}) with {best_acc_row['accuracy']:.4f}")
        print(f"  • Fastest Model : {fastest_row['model']} (k={fastest_row['k']}) in {fastest_row['training_time_mean']:.6f}s")

        if not topk2_row.empty:
            t2 = topk2_row.iloc[0]
            acc_diff = t2["accuracy"] - greedy_row["accuracy"]
            time_diff = t2["training_time_mean"] - greedy_row["training_time_mean"]
            sign_acc = "+" if acc_diff >= 0 else ""
            sign_time = "+" if time_diff >= 0 else ""
            print(f"  • Top-k (k=2) vs Greedy : Accuracy {sign_acc}{acc_diff:.4f}, Train Time {sign_time}{time_diff:.6f}s")

        if not topk3_row.empty:
            t3 = topk3_row.iloc[0]
            acc_diff = t3["accuracy"] - greedy_row["accuracy"]
            time_diff = t3["training_time_mean"] - greedy_row["training_time_mean"]
            sign_acc = "+" if acc_diff >= 0 else ""
            sign_time = "+" if time_diff >= 0 else ""
            print(f"  • Top-k (k=3) vs Greedy : Accuracy {sign_acc}{acc_diff:.4f}, Train Time {sign_time}{time_diff:.6f}s")


def main():
    print("=========================================================================")
    print(" Lightweight Top-k Decision Tree Learning — Harnessing the Power of Choices")
    print("=========================================================================")

    X, y = make_toy_dataset()
    print("\n1. Toy Dataset Sanity Checks:")
    train_and_report_toy(SimpleDecisionTree(max_depth=3), X, y, "Greedy (k=1)")
    train_and_report_toy(TopKDecisionTree(k=2, max_depth=3), X, y, "Top-k (k=2)")
    train_and_report_toy(TopKDecisionTree(k=3, max_depth=3), X, y, "Top-k (k=3)")

    print("\n2. Executing Full Benchmark Suite (Strict Train-Only Thresholding)...")
    results = run_experiments(max_depth=4, test_size=0.3, random_state=42, n_repetitions=3)

    print_results_table(results)
    print_summary_statistics(results)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    print("\n3. Outputs Generated Successfully:")
    print(f"  • CSV Files Saved : {os.path.join(root_dir, 'results', 'benchmark_scores.csv')}")
    print(f"                      {os.path.join(root_dir, 'experiments', 'results.csv')}")
    print(f"  • Graphs Saved to : {os.path.join(root_dir, 'results')}")
    print("    1. accuracy_comparison.png")
    print("    2. classification_metrics_comparison.png")
    print("    3. training_time_comparison.png")
    print("    4. prediction_time_comparison.png")
    print("    5. tree_depth_comparison.png")
    print("    6. node_count_comparison.png")
    print("    7. accuracy_vs_training_time.png")
    print("    8. top_k_effect.png")

    print("\nShort Summary:")
    print("  The Greedy Decision Tree selects the single best feature at each split via Information Gain.")
    print("  The Top-k Decision Tree evaluates candidate subtrees from the top-k features before choosing.")
    print("  This offers an explainable demonstration of how local search expansion can refine tree accuracy.")


if __name__ == "__main__":
    main()