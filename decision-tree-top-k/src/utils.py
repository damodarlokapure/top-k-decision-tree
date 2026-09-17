from __future__ import annotations

from collections import Counter

import numpy as np


def majority_class(y):
    values = list(y)
    if not values:
        return 0
    counter = Counter(values)
    max_count = max(counter.values())
    best_labels = [label for label, count in counter.items() if count == max_count]
    return int(min(best_labels))


def binarize_features(X, thresholds=None):
    X = np.asarray(X, dtype=float)
    if thresholds is None:
        thresholds = np.median(X, axis=0)
    binary_X = (X > thresholds).astype(int)
    return binary_X, np.asarray(thresholds)


def ensure_numpy_array(X):
    return np.asarray(X)
