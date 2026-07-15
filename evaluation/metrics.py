"""
All classification metrics the project brief requires, in one place:
accuracy, AUC-ROC, LogLoss, precision, recall, F1, and a confusion matrix.

Inputs are the model's raw logits (not probabilities) and integer labels.
We apply sigmoid here so callers never have to remember to. Threshold 0.5
turns probabilities into real/fake labels (1 = fake), per project convention.

AUC-ROC is the primary metric (threshold-free, robust to class imbalance);
the others describe behaviour at the 0.5 operating point. See glossary.md.
"""
import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, log_loss,
        precision_score, recall_score, f1_score, confusion_matrix, roc_curve,
    )
except ImportError as e:
    raise ImportError(f"scikit-learn required for metrics: {e}")


def _equal_error_rate(labels: np.ndarray, probs: np.ndarray) -> float:
    """
    EER = the error rate at the threshold where the false-positive rate equals
    the false-negative rate (FPR == 1 - TPR). Lower is better; it's a
    threshold-independent summary common in spoof/deepfake detection. We find
    the ROC point where |FPR - (1 - TPR)| is smallest and average the two.
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_metrics(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """
    logits, labels: 1-D arrays of equal length. Returns a dict of metrics.
    Guards the degenerate single-class case (AUC/LogLoss undefined then).
    """
    logits = np.asarray(logits, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    probs = 1.0 / (1.0 + np.exp(-logits))          # sigmoid -> P(fake)
    preds = (probs >= threshold).astype(np.int64)

    metrics = {}
    metrics["accuracy"] = float(accuracy_score(labels, preds))
    # zero_division=0: if the model predicts no positives, precision is 0 not NaN.
    metrics["precision"] = float(precision_score(labels, preds, zero_division=0))
    metrics["recall"] = float(recall_score(labels, preds, zero_division=0))
    metrics["f1"] = float(f1_score(labels, preds, zero_division=0))

    # AUC and LogLoss need both classes present; on a one-class batch they're
    # undefined -- report NaN rather than crash a validation pass.
    if len(np.unique(labels)) < 2:
        metrics["auc_roc"] = float("nan")
        metrics["log_loss"] = float("nan")
        metrics["eer"] = float("nan")
    else:
        metrics["auc_roc"] = float(roc_auc_score(labels, probs))
        metrics["log_loss"] = float(log_loss(labels, probs, labels=[0, 1]))
        metrics["eer"] = _equal_error_rate(labels, probs)

    cm = confusion_matrix(labels, preds, labels=[0, 1])   # rows=true, cols=pred
    metrics["confusion_matrix"] = cm.tolist()             # [[TN, FP], [FN, TP]]
    return metrics


def format_metrics(metrics: dict) -> str:
    """Human-readable one-liner + confusion matrix, for training logs."""
    cm = metrics["confusion_matrix"]
    line = (
        f"acc={metrics['accuracy']:.3f} auc={metrics['auc_roc']:.3f} "
        f"eer={metrics.get('eer', float('nan')):.3f} "
        f"logloss={metrics['log_loss']:.3f} P={metrics['precision']:.3f} "
        f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f}"
    )
    cm_str = (
        f"           pred_real pred_fake\n"
        f"true_real     {cm[0][0]:>6}    {cm[0][1]:>6}\n"
        f"true_fake     {cm[1][0]:>6}    {cm[1][1]:>6}"
    )
    return line + "\n" + cm_str


if __name__ == "__main__":
    # smoke test
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=100)
    logits = rng.normal(size=100) + (labels * 2 - 1)  # weakly correlated with truth
    print(format_metrics(compute_metrics(logits, labels)))
