from __future__ import annotations

import numpy as np
import pytest

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
from src.utils import binarize_features, majority_class


def toy_dataset():
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
        ],
        dtype=float,
    )
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=int)
    return X, y


# 1. Entropy tests
def test_entropy_zero_for_pure_node():
    tree = SimpleDecisionTree()
    assert tree.entropy(np.array([1, 1, 1])) == 0.0
    assert tree.entropy(np.array([0, 0, 0])) == 0.0


def test_entropy_known_value():
    tree = SimpleDecisionTree()
    value = tree.entropy(np.array([0, 0, 1, 1]))
    assert round(value, 5) == 1.0


def test_entropy_empty_dataset():
    tree = SimpleDecisionTree()
    assert tree.entropy([]) == 0.0


# 2. Information Gain tests
def test_information_gain_positive_on_useful_split():
    X, y = toy_dataset()
    tree = SimpleDecisionTree()
    gain = tree.information_gain(X, y, feature_index=2, split_value=1.0)
    assert gain >= 0.0


def test_information_gain_zero_on_invalid_split():
    X, y = toy_dataset()
    tree = SimpleDecisionTree()
    gain = tree.information_gain(X, y, feature_index=0, split_value=999.0)
    assert gain == 0.0


# 3. Feature ranking tests
def test_feature_ranking_greedy_and_top_k():
    X, y = toy_dataset()
    tree = SimpleDecisionTree()
    best_feat, best_val, gain = tree.get_best_feature(X, y)
    assert best_feat is not None
    assert gain > 0.0

    top_2 = tree.get_top_k_features(X, y, k=2)
    assert len(top_2) <= 2
    top_3 = tree.get_top_k_features(X, y, k=3)
    assert len(top_3) <= 3


# 4. Dataset splitting tests
def test_split_dataset_shapes():
    X, y = toy_dataset()
    tree = SimpleDecisionTree()
    left_X, left_y, right_X, right_y = tree.split_dataset(X, y, feature_index=0, split_value=1.0)
    assert len(left_X) + len(right_X) == len(X)
    assert len(left_y) + len(right_y) == len(y)


# 5. Binary Preprocessing & No Data Leakage
def test_binary_preprocessing_no_data_leakage():
    X_train_raw = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    X_test_raw = np.array([[0.0, 5.0], [4.0, 35.0]])

    X_train_bin, thresholds = binarize_features(X_train_raw)
    assert np.array_equal(thresholds, np.array([2.0, 20.0]))
    assert np.array_equal(X_train_bin, np.array([[0, 0], [0, 0], [1, 1]]))

    # Apply training thresholds to test data
    X_test_bin, _ = binarize_features(X_test_raw, thresholds=thresholds)
    assert np.array_equal(X_test_bin, np.array([[0, 0], [1, 1]]))


# 6. Model Training & Prediction
def test_model_training_and_prediction():
    X, y = toy_dataset()
    model = SimpleDecisionTree(max_depth=3)
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert calculate_accuracy(y, predictions) >= 0.75


# 7. Classification Metrics (Accuracy, Precision, Recall, F1)
def test_classification_metrics():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])

    acc = calculate_accuracy(y_true, y_pred)
    prec = calculate_precision(y_true, y_pred)
    rec = calculate_recall(y_true, y_pred)
    f1 = calculate_f1_score(y_true, y_pred)

    assert acc == 0.8
    assert 0.0 <= prec <= 1.0
    assert 0.0 <= rec <= 1.0
    assert 0.0 <= f1 <= 1.0


# 8. Tree Complexity Metrics
def test_tree_complexity_metrics():
    X, y = toy_dataset()
    model = SimpleDecisionTree(max_depth=3)
    model.fit(X, y)
    depth = calculate_tree_depth(model.root)
    nodes = count_nodes(model.root)
    leaves = count_leaves(model.root)

    assert 1 <= depth <= 4
    assert nodes >= 1
    assert leaves >= 1
    assert leaves <= nodes


# 9. Top-k behavior tests
def test_topk_k1_matches_greedy_behavior():
    X, y = toy_dataset()
    greedy = SimpleDecisionTree(max_depth=3)
    topk = TopKDecisionTree(k=1, max_depth=3)
    greedy.fit(X, y)
    topk.fit(X, y)
    assert np.array_equal(greedy.predict(X), topk.predict(X))


def test_topk_k2_and_k3_fit_on_toy_data():
    X, y = toy_dataset()
    model_k2 = TopKDecisionTree(k=2, max_depth=3)
    model_k3 = TopKDecisionTree(k=3, max_depth=3)
    model_k2.fit(X, y)
    model_k3.fit(X, y)
    assert model_k2.k == 2
    assert model_k3.k == 3
    assert len(model_k2.predict(X)) == len(y)
    assert len(model_k3.predict(X)) == len(y)


# 10. Edge Cases: Empty or pure datasets & repeated feature values
def test_pure_dataset():
    X = np.array([[0, 1], [0, 1], [0, 1]])
    y = np.array([1, 1, 1])
    model = SimpleDecisionTree(max_depth=3)
    model.fit(X, y)
    assert np.array_equal(model.predict(X), y)


def test_repeated_feature_values():
    X = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
    y = np.array([0, 1, 0, 1])
    model = SimpleDecisionTree(max_depth=3)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert np.all(preds == majority_class(y))