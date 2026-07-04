# Shared Glossary

**Owner:** Person 1 (Research lead), Day 1 deliverable.
Written for the whole team — plain language first, formulas only where they help.

## Data splits

### Training / validation / test splits

<!-- What each split is for, why the test set must stay untouched until the end,
     and (for this project) why splits must be identity-disjoint — see
     PROJECT_OVERVIEW.md §5. -->

## Training concepts

### Loss function

<!-- What a loss measures, and the one we'll use most: binary cross-entropy
     (same thing as LogLoss below). -->

### Overfitting

<!-- Model memorizes training data instead of learning the pattern; how to
     spot it (train vs validation curves diverge) and common fixes. -->

## Metrics

### Accuracy

<!-- Fraction correct — and why it's misleading on imbalanced data like
     FakeAVCeleb (predicting "fake" for everything scores ~97%). -->

### AUC-ROC

<!-- Probability the model ranks a random fake above a random real.
     Threshold-free, robust to imbalance — our primary metric. -->

### LogLoss

<!-- Penalizes confident wrong answers; rewards well-calibrated probabilities.
     Matters because fusion consumes each stream's probability, not just its verdict. -->

### Confusion matrix

<!-- The 2x2 table: true/false positives/negatives. Added to the training
     notebook on Day 3. -->
