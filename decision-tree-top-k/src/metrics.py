from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def calculate_precision(y_true, y_pred) -> float:
    return float(precision_score(y_true, y_pred, average="weighted", zero_division=0))


def calculate_recall(y_true, y_pred) -> float:
    return float(recall_score(y_true, y_pred, average="weighted", zero_division=0))


def calculate_f1_score(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def calculate_tree_depth(node) -> int:
    if node is None:
        return 0
    if node.is_leaf:
        return 1
    return 1 + max(calculate_tree_depth(node.left), calculate_tree_depth(node.right))


def count_nodes(node) -> int:
    if node is None:
        return 0
    if node.is_leaf:
        return 1
    return 1 + count_nodes(node.left) + count_nodes(node.right)


def count_leaves(node) -> int:
    if node is None:
        return 0
    if node.is_leaf:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)

