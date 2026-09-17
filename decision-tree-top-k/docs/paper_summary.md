# Paper Summary: Harnessing the Power of Choices in Decision Tree Learning

## Problem Addressed

The paper studies a weakness of classic greedy decision tree learning. Standard algorithms such as ID3, C4.5, and CART usually choose only the single best feature at each split. That is fast and simple, but it can miss better trees because the locally best split is not always the globally best choice.

## Main Idea

The paper proposes Top-k decision tree learning. Instead of picking only the best feature, the algorithm looks at the top k candidate features at each node and then chooses among those options.

The key idea is simple:

- `k = 1` means fully greedy learning.
- Larger `k` means the algorithm has more choices at each node.

## Normal Greedy Decision Tree Approach

In a greedy tree, the algorithm:

1. Scores every feature.
2. Chooses the single best feature.
3. Splits the data on that feature.
4. Repeats the process on the child nodes.

This is fast and easy to understand, but it can get stuck with a split that looks good locally and hurts the tree later.

## Top-k Decision Tree Approach

Top-k keeps the same basic tree structure, but it does not commit to only one feature immediately.

At each node, it:

1. Scores all available features.
2. Keeps the top k features.
3. Builds a candidate tree for each of those features.
4. Compares the candidate trees.
5. Chooses the best one.

This gives the learner a small number of extra choices without moving to a heavy optimization method.

## Simple Example

Suppose three features have scores:

- Feature A: 0.40
- Feature B: 0.39
- Feature C: 0.20

A greedy tree chooses A immediately.

A Top-2 tree considers A and B. If B leads to a better subtree, Top-2 may choose B instead.

That is the main benefit of the method: a slightly larger search can produce a better tree.

## Advantages

- Still easy to explain.
- Keeps the familiar decision tree structure.
- Can improve accuracy compared with pure greedy learning.
- Uses only a small number of extra choices.

## Limitations

- It is still a heuristic, not a guaranteed optimal tree.
- Increasing k increases training cost.
- The paper includes theoretical results and larger experiments that are not reproduced fully here.

## What This Project Implements

This project implements a lightweight student-level version of the main idea:

- entropy and information gain
- a manual greedy decision tree classifier
- a simplified Top-k tree classifier
- toy and small benchmark datasets
- accuracy, depth, node count, and timing experiments
- plots and CSV outputs

## What This Project Does Not Implement

This project intentionally does not implement:

- the full theoretical proofs from the paper
- the full-scale experimental setup from the paper
- large dataset benchmarking
- advanced optimization methods or specialized solvers
- the exact research codebase used by the authors
