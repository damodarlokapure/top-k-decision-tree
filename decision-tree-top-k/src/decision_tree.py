from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from .metrics import calculate_accuracy
from .utils import majority_class


@dataclass
class TreeNode:
    """Represents a node in the decision tree."""
    is_leaf: bool
    prediction: int
    feature_index: Optional[int] = None
    split_value: Optional[Any] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    depth: int = 1
    num_samples: int = 0
    impurity: float = 0.0
    gain: float = 0.0


class SimpleDecisionTree:
    """
    Model 1 — Greedy Decision Tree Classifier.

    At every node:
    1. Calculate information gain for all available features.
    2. Select the feature with the highest information gain.
    3. Split the dataset recursively.
    """

    def __init__(self, max_depth: int = 3, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[TreeNode] = None
        self.n_features_: Optional[int] = None

    def entropy(self, y: Sequence[int]) -> float:
        """Calculate binary / multi-class Shannon entropy using base 2."""
        values = np.asarray(y)
        if len(values) == 0:
            return 0.0
        _, counts = np.unique(values, return_counts=True)
        probabilities = counts / counts.sum()
        ent = 0.0
        for p in probabilities:
            if p > 0:
                ent -= p * np.log2(p)
        return float(ent)

    def _split_dataset(self, X, y, feature_index: int, split_value: Any):
        """Split dataset into matching (left) and non-matching (right) subsets."""
        X = np.asarray(X)
        y = np.asarray(y)
        mask = X[:, feature_index] == split_value
        return X[mask], y[mask], X[~mask], y[~mask]

    def split_dataset(self, X, y, feature_index: int, split_value: Any):
        return self._split_dataset(X, y, feature_index, split_value)

    def information_gain(self, X, y, feature_index: int, split_value: Any) -> float:
        """
        Calculate information gain for a candidate split:
        gain = parent_entropy - weighted_child_entropy
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if len(y) == 0:
            return 0.0
        left_X, left_y, right_X, right_y = self._split_dataset(X, y, feature_index, split_value)
        if len(left_y) == 0 or len(right_y) == 0:
            return 0.0
        parent_entropy = self.entropy(y)
        left_entropy = self.entropy(left_y)
        right_entropy = self.entropy(right_y)
        weighted_child_entropy = (len(left_y) / len(y)) * left_entropy + (len(right_y) / len(y)) * right_entropy
        return float(parent_entropy - weighted_child_entropy)

    def _get_feature_candidates(self, X, y, available_features: Sequence[int]):
        """Evaluate candidate splits for all available features and return them sorted by gain."""
        candidates = []
        for feature_index in available_features:
            unique_values = np.unique(X[:, feature_index])
            best_value = None
            best_gain = -1.0
            for split_value in unique_values:
                gain = self.information_gain(X, y, feature_index, split_value)
                if gain > best_gain:
                    best_gain = gain
                    best_value = split_value
            if best_value is not None:
                candidates.append((feature_index, best_value, float(best_gain)))
        candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
        return candidates

    def get_best_feature(self, X, y, available_features: Optional[Sequence[int]] = None):
        """Return the single best feature split according to Information Gain."""
        X = np.asarray(X)
        y = np.asarray(y)
        if available_features is None:
            available_features = list(range(X.shape[1]))
        candidates = self._get_feature_candidates(X, y, available_features)
        if not candidates:
            return None, None, 0.0
        feature_index, split_value, gain = candidates[0]
        return feature_index, split_value, gain

    def get_top_k_features(self, X, y, available_features: Optional[Sequence[int]] = None, k: int = 1):
        """Return the top k feature splits ranked by Information Gain."""
        X = np.asarray(X)
        y = np.asarray(y)
        if available_features is None:
            available_features = list(range(X.shape[1]))
        candidates = self._get_feature_candidates(X, y, available_features)
        return candidates[:k]

    def _should_stop(self, y, depth_remaining: int, available_features: Sequence[int]):
        """Check stopping conditions: empty data, pure node, depth limit, no features, or min samples limit."""
        if len(y) == 0:
            return True
        if len(np.unique(y)) <= 1:
            return True
        if depth_remaining <= 1:
            return True
        if len(available_features) == 0:
            return True
        if len(y) < self.min_samples_split:
            return True
        return False

    def _choose_split(self, X, y, available_features: Sequence[int], depth_remaining: int):
        """Greedy decision strategy: select the feature with highest Information Gain."""
        return self.get_best_feature(X, y, available_features)

    def _build_tree(self, X, y, depth_remaining: int, available_features: Sequence[int], forced_split=None):
        """Recursively construct the decision tree."""
        X = np.asarray(X)
        y = np.asarray(y)
        majority = majority_class(y)

        if self._should_stop(y, depth_remaining, available_features):
            return TreeNode(is_leaf=True, prediction=majority, depth=1, num_samples=len(y), impurity=self.entropy(y))

        if forced_split is None:
            feature_index, split_value, gain = self._choose_split(X, y, available_features, depth_remaining)
        else:
            feature_index, split_value, gain = forced_split

        if feature_index is None:
            return TreeNode(is_leaf=True, prediction=majority, depth=1, num_samples=len(y), impurity=self.entropy(y))

        left_X, left_y, right_X, right_y = self._split_dataset(X, y, feature_index, split_value)
        if len(left_y) == 0 or len(right_y) == 0 or gain <= 0:
            return TreeNode(is_leaf=True, prediction=majority, depth=1, num_samples=len(y), impurity=self.entropy(y))

        remaining_features = [feature for feature in available_features if feature != feature_index]
        left_child = self._build_tree(left_X, left_y, depth_remaining - 1, remaining_features)
        right_child = self._build_tree(right_X, right_y, depth_remaining - 1, remaining_features)

        node_depth = 1 + max(left_child.depth, right_child.depth)
        return TreeNode(
            is_leaf=False,
            prediction=majority,
            feature_index=feature_index,
            split_value=split_value,
            left=left_child,
            right=right_child,
            depth=node_depth,
            num_samples=len(y),
            impurity=self.entropy(y),
            gain=gain,
        )

    def fit(self, X, y):
        """Fit the decision tree on dataset X and target y."""
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        self.n_features_ = X.shape[1]
        available_features = list(range(self.n_features_))
        self.root = self._build_tree(X, y, self.max_depth, available_features)
        return self

    def predict_one(self, x, node: Optional[TreeNode] = None):
        """Predict target class for a single feature vector."""
        if node is None:
            node = self.root
        if node is None:
            raise ValueError("The model has not been fitted yet")
        if node.is_leaf or node.feature_index is None:
            return node.prediction
        
        target_child = node.left if x[node.feature_index] == node.split_value else node.right
        if target_child is None:
            return node.prediction
        return self.predict_one(x, target_child)

    def predict(self, X):
        """Predict target classes for all rows in matrix X."""
        X = np.asarray(X)
        return np.array([self.predict_one(row) for row in X])

    def score(self, X, y):
        """Calculate classification accuracy on test dataset."""
        return calculate_accuracy(y, self.predict(X))


class TopKDecisionTree(SimpleDecisionTree):
    """
    Model 2 — Top-k Decision Tree Classifier.

    At every node:
    1. Calculate information gain for all available features.
    2. Select top k candidate features.
    3. Build candidate subtrees for each choice.
    4. Select the candidate split yielding highest local training accuracy,
       using Information Gain as a tie-breaker.
    
    Note: This uses a simplified practical approximation (local node training accuracy)
    rather than full global validation or theoretical optimization from the paper.
    """

    def __init__(self, k: int = 2, max_depth: int = 3, min_samples_split: int = 2):
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split)
        self.k = max(1, int(k))

    def _choose_split(self, X, y, available_features: Sequence[int], depth_remaining: int):
        candidates = self.get_top_k_features(X, y, available_features, k=self.k)
        if not candidates:
            return None, None, 0.0
        if len(candidates) == 1:
            return candidates[0]

        best_candidate = None
        best_accuracy = -1.0
        best_gain = -1.0

        for candidate in candidates:
            # Build candidate subtree starting from current node
            candidate_tree = self._build_tree(X, y, depth_remaining, available_features, forced_split=candidate)
            predicted = self._predict_with_tree(X, candidate_tree)
            accuracy = calculate_accuracy(y, predicted)
            
            # Select candidate with highest local accuracy; use Information Gain as tie-breaker
            if accuracy > best_accuracy or (np.isclose(accuracy, best_accuracy) and candidate[2] > best_gain):
                best_candidate = candidate
                best_accuracy = accuracy
                best_gain = candidate[2]

        return best_candidate if best_candidate is not None else candidates[0]

    def _predict_with_tree(self, X, node):
        X = np.asarray(X)
        return np.array([self.predict_one(row, node=node) for row in X])