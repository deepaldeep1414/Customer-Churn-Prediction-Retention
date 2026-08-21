"""
ml/evaluation/threshold_selector.py
Capacity-constrained threshold selection.

The decision threshold is selected on the VALIDATION SET ONLY and then
frozen. It is never re-adjusted on the test set.

Strategy: Find the highest threshold (→ highest precision) such that
the number of predicted positives does not exceed retention_capacity.
This ties the model's operating point directly to the business constraint.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import precision_recall_curve

logger = logging.getLogger(__name__)


def select_threshold(
    y_val: np.ndarray,
    y_prob_val: np.ndarray,
    retention_capacity: int,
) -> float:
    """
    Select decision threshold tied to retention team capacity.

    Scans thresholds from high → low. Returns the first threshold
    where predicted positives <= retention_capacity, maximising
    recall within that constraint.

    If no threshold satisfies the capacity constraint, returns the
    threshold that minimises the number of predicted positives (highest
    possible threshold).

    Args:
        y_val:              Ground-truth labels (0/1) on validation set.
        y_prob_val:         Predicted probabilities on validation set.
        retention_capacity: Maximum number of customers the retention
                            team can handle per month.

    Returns:
        Frozen threshold (float between 0 and 1).
    """
    y_val     = np.asarray(y_val)
    y_prob_val = np.asarray(y_prob_val)

    _, _, thresholds = precision_recall_curve(y_val, y_prob_val)
    # thresholds are in ascending order; iterate descending (most selective first)
    thresholds_desc = np.sort(thresholds)[::-1]

    selected = float(thresholds_desc[0])   # fallback: most conservative

    for thresh in thresholds_desc:
        predicted_positives = int((y_prob_val >= thresh).sum())
        if predicted_positives <= retention_capacity:
            selected = float(thresh)
            break

    predicted_at_selected = int((y_prob_val >= selected).sum())
    logger.info(
        "Threshold selected: %.4f | predicted_positives=%d | capacity=%d",
        selected, predicted_at_selected, retention_capacity,
    )

    if predicted_at_selected > retention_capacity:
        logger.warning(
            "Could not satisfy capacity constraint (%d). "
            "Using highest threshold: %.4f (predicts %d positives).",
            retention_capacity, selected, predicted_at_selected,
        )

    return selected


def threshold_sweep(
    y_val: np.ndarray,
    y_prob_val: np.ndarray,
    n_points: int = 50,
) -> list[dict]:
    """
    Sweep threshold from 0.01 to 0.99 and return metrics at each point.
    Useful for plotting precision-recall-threshold curves.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    y_val     = np.asarray(y_val)
    y_prob_val = np.asarray(y_prob_val)
    results   = []

    for thresh in np.linspace(0.01, 0.99, n_points):
        y_pred = (y_prob_val >= thresh).astype(int)
        results.append({
            "threshold":         round(float(thresh), 4),
            "predicted_positives": int(y_pred.sum()),
            "recall":            round(float(recall_score(y_val, y_pred, zero_division=0)), 4),
            "precision":         round(float(precision_score(y_val, y_pred, zero_division=0)), 4),
            "f1":                round(float(f1_score(y_val, y_pred, zero_division=0)), 4),
        })

    return results
