"""
ml/evaluation/evaluator.py
Shared evaluation protocol used by ALL models identically.
No model-specific branching here — ensures apples-to-apples comparison.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalResults:
    prefix: str
    pr_auc:    float
    roc_auc:   float
    brier:     float
    recall:    float
    precision: float
    f1:        float
    threshold: float
    ece:       float | None = None  # Expected Calibration Error

    def as_dict(self) -> dict[str, float]:
        """Return flat dict suitable for mlflow.log_metrics()."""
        return {
            f"{self.prefix}_pr_auc":    round(self.pr_auc, 5),
            f"{self.prefix}_roc_auc":   round(self.roc_auc, 5),
            f"{self.prefix}_brier":     round(self.brier, 5),
            f"{self.prefix}_recall":    round(self.recall, 5),
            f"{self.prefix}_precision": round(self.precision, 5),
            f"{self.prefix}_f1":        round(self.f1, 5),
            **({f"{self.prefix}_ece": round(self.ece, 5)} if self.ece is not None else {}),
        }

    def summary(self) -> str:
        return (
            f"[{self.prefix}] PR-AUC={self.pr_auc:.4f} | ROC-AUC={self.roc_auc:.4f} | "
            f"Brier={self.brier:.4f} | Recall={self.recall:.4f} | "
            f"Precision={self.precision:.4f} | F1={self.f1:.4f} | "
            f"Threshold={self.threshold:.4f}"
            + (f" | ECE={self.ece:.4f}" if self.ece is not None else "")
        )


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).
    Lower is better; 0.0 = perfectly calibrated.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    prefix: str = "eval",
    compute_ece: bool = True,
) -> EvalResults:
    """
    Compute the full shared evaluation suite.

    Args:
        y_true:    Ground-truth binary labels (0/1).
        y_prob:    Predicted churn probabilities (0–1).
        threshold: Decision threshold for binary classification metrics.
        prefix:    Metric name prefix for MLflow logging.
        compute_ece: Whether to compute Expected Calibration Error.

    Returns:
        EvalResults dataclass.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc    = float(average_precision_score(y_true, y_prob))
    roc_auc   = float(roc_auc_score(y_true, y_prob))
    brier     = float(brier_score_loss(y_true, y_prob))
    recall    = float(recall_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    f1        = float(f1_score(y_true, y_pred, zero_division=0))
    ece       = expected_calibration_error(y_true, y_prob) if compute_ece else None

    result = EvalResults(
        prefix=prefix,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        brier=brier,
        recall=recall,
        precision=precision,
        f1=f1,
        threshold=threshold,
        ece=ece,
    )
    logger.info(result.summary())
    return result


# ── Promotion gate checks ─────────────────────────────────────
GATE_PR_AUC_MIN         = 0.75   # Absolute floor
GATE_PR_AUC_DELTA       = -0.02  # Allowed drop vs champion
GATE_ECE_MAX            = 0.05   # Calibration gate
GATE_RECALL_DELTA       = -0.03  # Allowed drop vs champion


def check_promotion_gates(
    candidate: EvalResults,
    champion: EvalResults | None,
) -> tuple[bool, list[str]]:
    """
    Run all automated promotion gates.
    Returns (passed: bool, failures: list[str]).
    """
    failures: list[str] = []

    # Gate 1: Absolute PR-AUC floor
    if candidate.pr_auc < GATE_PR_AUC_MIN:
        failures.append(
            f"PR-AUC {candidate.pr_auc:.4f} < floor {GATE_PR_AUC_MIN}"
        )

    # Gate 2: ECE calibration
    if candidate.ece is None:
        failures.append("ECE is None — calibration step was skipped")
    elif candidate.ece > GATE_ECE_MAX:
        failures.append(
            f"ECE {candidate.ece:.4f} > max {GATE_ECE_MAX}"
        )

    if champion is not None:
        # Gate 3: PR-AUC regression vs champion
        delta_pr = candidate.pr_auc - champion.pr_auc
        if delta_pr < GATE_PR_AUC_DELTA:
            failures.append(
                f"PR-AUC delta vs champion: {delta_pr:+.4f} < {GATE_PR_AUC_DELTA}"
            )

        # Gate 4: Recall regression vs champion
        delta_recall = candidate.recall - champion.recall
        if delta_recall < GATE_RECALL_DELTA:
            failures.append(
                f"Recall delta vs champion: {delta_recall:+.4f} < {GATE_RECALL_DELTA}"
            )

    passed = len(failures) == 0
    if passed:
        logger.info("✅ All promotion gates PASSED")
    else:
        for f in failures:
            logger.warning("❌ Gate FAILED: %s", f)

    return passed, failures
