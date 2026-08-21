"""
ml/evaluation/calibration.py
Post-training probability calibration.
Calibrator is fitted on the validation set ONLY (cv='prefit').
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def calibrate_model(
    pipeline: Pipeline,
    X_val,
    y_val,
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    """
    Wrap an already-fitted pipeline in a calibrator fitted on val set.

    Args:
        pipeline: Already-fitted sklearn Pipeline.
        X_val, y_val: Validation features and labels.
        method: 'isotonic' (default, non-parametric, best for n_val > 1000)
                or 'sigmoid' (Platt scaling, better for very small val sets).

    Returns:
        CalibratedClassifierCV fitted on val set.

    Note: CalibratedClassifierCV with cv='prefit' fits the calibrator
    on (X_val, y_val) without refitting the base estimator.
    """
    n_val = len(y_val)
    if n_val < 200 and method == "isotonic":
        logger.warning(
            "Val set size=%d is small for isotonic calibration. "
            "Consider method='sigmoid'.", n_val
        )

    bincounts = np.bincount(y_val) if len(y_val) > 0 else np.array([])
    min_class = int(np.min(bincounts)) if len(bincounts) > 1 else 5
    cv_splits = max(2, min(5, min_class)) if min_class >= 2 else 2

    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(pipeline),
            method=method,
            cv=cv_splits,
        )
    except (ImportError, AttributeError):
        calibrated = CalibratedClassifierCV(
            estimator=pipeline,
            method=method,
            cv="prefit",
        )
    calibrated.fit(X_val, y_val)
    logger.info(
        "Calibration complete | method=%s | n_val=%d", method, n_val
    )
    return calibrated


def reliability_curve_data(
    y_true,
    y_prob,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (mean_predicted_value, fraction_of_positives) for reliability diagram.
    """
    fraction_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    return mean_pred, fraction_pos
