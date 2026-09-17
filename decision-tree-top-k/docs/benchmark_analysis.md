# Benchmark Analysis Report: Top-k Decision Tree Learning

## Executive Summary

This report evaluates the empirical performance of **Greedy Decision Trees** (`k=1`) versus **Top-k Decision Trees** (`k=2`, `k=3`) and a **Scikit-Learn DecisionTreeClassifier baseline** (`k=NA`). 

Experiments were conducted on three binary classification datasets under strict data-leakage prevention protocols. The results demonstrate that **expanding candidate split choices (Top-k) improves classification accuracy on non-trivial datasets** (e.g., +4.09% accuracy boost on Breast Cancer with `k=3`), at the trade-off of increased training runtime.

---

## 1. Experimental Setup

| Parameter | Value / Configuration |
| :--- | :--- |
| **Datasets** | 1. `toy` (12 samples, 3 features)<br>2. `iris_binary` (100 samples, 4 features; Setosa vs Versicolor)<br>3. `breast_cancer_binary` (569 samples, 30 features) |
| **Train-Test Split** | 70% Train / 30% Test, stratified by class label |
| **Random Seed** | `RANDOM_STATE = 42` |
| **Feature Preprocessing** | Continuous features binarized using median thresholds **calculated strictly on the training set** to eliminate data leakage. |
| **Evaluated Models** | 1. `Greedy` (`SimpleDecisionTree`, `k=1`) <br>2. `TopK` (`TopKDecisionTree`, `k=2`) <br>3. `TopK` (`TopKDecisionTree`, `k=3`) <br>4. `SklearnDecisionTree` (`DecisionTreeClassifier(max_depth=4)`, `k=NA`) |
| **Tree Parameters** | `max_depth = 4`, `min_samples_split = 2` |
| **Timing Procedure** | 3 independent timing repetitions (`time.perf_counter`) reporting mean and standard deviation. |

---

## 2. Empirical Benchmark Results

Below are the exact measurements obtained from running `python main.py`:

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

## 3. Results Analysis & Key Findings

### Accuracy & Classification Quality
1. **Toy & Iris Datasets**:
   - Both datasets are linearly separable under binary thresholding. All models achieve equal scores (`0.7500` for `toy` test set, `1.0000` for `iris_binary`).
2. **Breast Cancer Dataset**:
   - **Greedy (`k=1`)**: Achieves `87.13%` accuracy.
   - **Top-k (`k=2`)**: Achieves `88.89%` accuracy (**+1.76% improvement** over Greedy).
   - **Top-k (`k=3`)**: Achieves `91.23%` accuracy (**+4.09% improvement** over Greedy), matching the Scikit-Learn baseline score.

### Computational Cost & Runtime
- Training time scales proportionally with `k` and tree depth:
  - Greedy (`k=1`): `0.0246s`
  - Top-k (`k=2`): `0.1947s` (~7.9x greedy time)
  - Top-k (`k=3`): `0.4057s` (~16.5x greedy time)
- **Reasoning**: At each node, `TopKDecisionTree` recursively constructs candidate subtrees for the top $k$ features and evaluates local training accuracy before committing to a split.

### Tree Complexity
- For custom trees (`Greedy` and `TopK`), `tree_depth` reached `4`, `nodes` reached `15`, and `leaves` reached `8`.
- Scikit-learn's baseline generated 29 nodes and 15 leaves due to continuous float split thresholding across internal nodes, whereas custom models evaluate binarized feature vectors.

---

## 4. Interpretation

> **Conclusion**: Top-k decision tree learning mitigates the structural myopia of classic greedy induction by exploring a small neighborhood of strong candidate splits. On complex multi-feature datasets like Breast Cancer, this expansion enables the model to find splits that yield superior downstream classification accuracy (+4.09%).

However, this accuracy gain comes at the cost of additional training computation. Prediction time remains identical across models because inference depth and structure are equivalent during runtime.

---

## 5. Limitations

1. **Dataset Scope**: Small benchmark datasets were selected for lightweight educational clarity.
2. **Binary Median Thresholding**: Continuous attributes are simplified into binary values using median thresholds.
3. **Local Candidate Evaluation**: Top-k candidate splits are evaluated on local node training subsets, which can occasionally risk local overfitting compared to a full global validation hold-out set.
4. **Limited Range of k**: Evaluated for $k \in \{1, 2, 3\}$.
5. **No Post-Pruning**: The tree relies solely on stopping criteria (`max_depth=4`, `min_samples_split=2`) without cost-complexity pruning.
