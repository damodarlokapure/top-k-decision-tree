# Project Refinement Plan: Top-k Decision Tree Learning

## 1. Current Implementation Summary
The current workspace contains a lightweight educational implementation of Top-k Decision Tree Learning inspired by the paper *"Harnessing the Power of Choices in Decision Tree Learning"*.
- Core classes: `SimpleDecisionTree` (Greedy, `k=1`) and `TopKDecisionTree` (`k>=1`) in `src/decision_tree.py`.
- Support scripts: `src/utils.py`, `src/metrics.py`, `experiments/run_experiments.py`, `main.py`, `tests/test_models.py`.
- Documentation: `docs/paper_summary.md`, `docs/implementation_notes.md`, `README.md`.

## 2. Problems Found
1. **Data Leakage in Preprocessing**: Feature binarization median thresholds were computed on the full dataset prior to `train_test_split`. Thresholds must be computed exclusively on training data and applied to both train and test splits.
2. **Incomplete Benchmark Metrics**: Precision, recall, F1-score, prediction time, timing standard deviations, leaf count, and scikit-learn baseline model were missing from the benchmark suite.
3. **Graph Limitations**: Only 2 graphs were generated (`accuracy_comparison.png` and `depth_comparison.png`). Eight distinct graphs are required.
4. **Stopping Condition & Tree Depth Calculation**: Depth calculation and depth-remaining stopping checks in `_build_tree` need refinement to ensure exact tree depth limits.
5. **Test Coverage**: Test suite lacked checks for data leakage, precision/recall/F1, leaf counts, edge cases (empty/pure datasets), and explicit `k=1, 2, 3` feature choices.

## 3. Changes Required
- **Data Leakage Fix**: Update `binarize_features` in `src/utils.py` and data preparation in `experiments/run_experiments.py` to fit median thresholds strictly on `X_train`.
- **Algorithm & Model Safety**: Refine `SimpleDecisionTree` and `TopKDecisionTree` depth tracking, stopping rules, leaf node creation, prediction fallbacks, and candidate evaluation docstrings.
- **Metrics Expansion**: Update `src/metrics.py` to include weighted precision, recall, F1-score, accuracy, depth, node count, and leaf count.
- **Baseline Integration**: Add `SklearnDecisionTree` (`DecisionTreeClassifier(max_depth=4, random_state=42)`) as an external reference model.

## 4. Benchmark Improvements
- Use fixed seed (`RANDOM_STATE = 42`) and identical train-test splits for all models on each dataset (`toy`, `iris_binary`, `breast_cancer_binary`).
- Execute 3 timing repetitions per model to compute mean and standard deviation for training time and prediction time.
- Generate standard CSV outputs: `results/benchmark_scores.csv` and `experiments/results.csv`.

## 5. Graphs to Generate
All 8 graphs will be saved in `results/`:
1. `accuracy_comparison.png`
2. `classification_metrics_comparison.png`
3. `training_time_comparison.png`
4. `prediction_time_comparison.png`
5. `tree_depth_comparison.png`
6. `node_count_comparison.png`
7. `accuracy_vs_training_time.png`
8. `top_k_effect.png`

## 6. Testing Plan
- Expand `tests/test_models.py` to cover entropy, information gain, feature selection, data leakage prevention, Greedy vs Top-k behavior (`k=1,2,3`), leaf counting, and edge cases.
- Execute `python -m pytest` and `python main.py` to verify all components end-to-end.
