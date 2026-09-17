# Implementation Notes

## Entropy

Entropy measures how mixed the labels are at a node.

For binary labels, the formula is:

```text
entropy(y) = -p0 log2(p0) - p1 log2(p1)
```

where `p0` and `p1` are the proportions of the two classes.

If all labels are the same, entropy is 0.

## Information Gain

Information gain measures how much a split reduces entropy.

In simple form:

```text
gain = parent_entropy - weighted_child_entropy
```

A better split gives a larger information gain.

## Normal Greedy Tree

The greedy tree works like this:

1. Score all available features.
2. Pick the best one.
3. Split the data.
4. Repeat on the child nodes.

This is `k = 1` behavior.

## Top-k Tree

The Top-k tree first ranks the features by information gain.

Then it keeps only the top `k` features and evaluates candidate trees built from those options.

In this project, the candidate tree is built recursively and the best candidate is selected using training accuracy on the current node data.

## What Happens When k = 1

When `k = 1`, there is only one candidate split. That makes the Top-k model behave like the greedy model.

## What Happens When k = 2 or k = 3

When `k = 2` or `k = 3`, the algorithm has more choices.

That extra choice can help it avoid a locally good split that leads to a weaker subtree.

## Dataset Preprocessing

The project uses small datasets:

- a manual toy dataset
- the Iris dataset filtered to two classes and binarized
- the Breast Cancer dataset binarized with median thresholds

Continuous features are converted to binary values using simple median thresholds so the tree implementation stays easy to explain.

## Experiments

For each dataset, the project:

1. Creates one train-test split.
2. Trains the greedy tree.
3. Trains Top-k with `k = 2`.
4. Trains Top-k with `k = 3`.
5. Records accuracy, training time, tree depth, and number of nodes.

The results are saved to CSV files and plotted with Matplotlib.

## Why This Is Lighter Than the Original Paper

This project is lighter because it:

- uses small datasets
- uses a simple recursive tree builder
- uses only NumPy, Pandas, Matplotlib, and scikit-learn utilities
- avoids specialized optimization code
- does not attempt to reproduce all theoretical results

## How to Explain It in a Viva

You can explain the project in three steps:

1. A decision tree chooses a feature at each node using information gain.
2. Greedy learning always picks the best immediate feature.
3. Top-k learning looks at a few strong candidates before deciding, which can improve the final tree.
