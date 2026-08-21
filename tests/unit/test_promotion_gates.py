"""
tests/unit/test_promotion_gates.py
Unit tests that verify every promotion gate blocks the right candidates.

Key test: promotion gates must BLOCK bad candidates — not silently pass.
After F-05 fix, missing ECE (val_cal_ece=None) must also fail G2.
"""
from __future__ import annotations

import pytest
import numpy as np
from ml.evaluation.evaluator import (
    EvalResults,
    check_promotion_gates,
    GATE_PR_AUC_MIN,
    GATE_ECE_MAX,
    GATE_PR_AUC_DELTA,
    GATE_RECALL_DELTA,
)


def _make_eval(
    pr_auc: float = 0.82,
    ece: float | None = 0.03,
    recall: float = 0.76,
    roc_auc: float = 0.88,
    precision: float = 0.67,
    f1: float = 0.71,
    brier: float = 0.12,
    threshold: float = 0.50,
    prefix: str = "test",
) -> EvalResults:
    return EvalResults(
        prefix=prefix, pr_auc=pr_auc, roc_auc=roc_auc, brier=brier,
        recall=recall, precision=precision, f1=f1, threshold=threshold, ece=ece,
    )


# ── G1: Absolute PR-AUC floor ────────────────────────────────

class TestG1AbsolutePRAUCFloor:

    def test_passes_above_floor(self):
        candidate = _make_eval(pr_auc=GATE_PR_AUC_MIN + 0.01)
        passed, failures = check_promotion_gates(candidate, champion=None)
        assert passed
        assert not any("PR-AUC" in f and "floor" in f.lower() for f in failures)

    def test_fails_at_exact_floor(self):
        # Exactly at floor — should FAIL (strict <)
        candidate = _make_eval(pr_auc=GATE_PR_AUC_MIN - 0.001)
        passed, failures = check_promotion_gates(candidate, champion=None)
        assert not passed
        assert any("PR-AUC" in f for f in failures)

    def test_fails_below_floor(self):
        candidate = _make_eval(pr_auc=0.60)
        passed, failures = check_promotion_gates(candidate, champion=None)
        assert not passed
        assert any("PR-AUC" in f for f in failures)

    @pytest.mark.parametrize("pr_auc", [0.40, 0.60, 0.74])
    def test_fails_various_low_pr_aucs(self, pr_auc: float):
        candidate = _make_eval(pr_auc=pr_auc)
        passed, _ = check_promotion_gates(candidate, champion=None)
        assert not passed, f"pr_auc={pr_auc} should have failed G1"


# ── G2: ECE calibration gate ─────────────────────────────────

class TestG2CalibrationGate:

    def test_passes_low_ece(self):
        candidate = _make_eval(ece=0.01)
        passed, failures = check_promotion_gates(candidate, champion=None)
        assert passed

    def test_fails_high_ece(self):
        candidate = _make_eval(ece=GATE_ECE_MAX + 0.01)
        passed, failures = check_promotion_gates(candidate, champion=None)
        assert not passed
        assert any("ECE" in f for f in failures)

    def test_fails_when_ece_is_none(self):
        """
        F-05: Missing ECE (calibration step skipped) must be a gate failure.
        This test will FAIL on the old code and PASS after F-05 fix.
        """
        candidate = _make_eval(ece=None)
        passed, failures = check_promotion_gates(candidate, champion=None)
        # After F-05 fix in evaluator.py:
        assert not passed, (
            "G2 must FAIL when ECE is None (calibration step skipped). "
            "Apply F-05 fix to check_promotion_gates()."
        )
        assert any("ECE" in f or "calibration" in f.lower() for f in failures), (
            f"Failure messages should mention ECE or calibration: {failures}"
        )

    @pytest.mark.parametrize("ece", [0.06, 0.10, 0.50])
    def test_fails_various_high_eces(self, ece: float):
        candidate = _make_eval(ece=ece)
        passed, _ = check_promotion_gates(candidate, champion=None)
        assert not passed, f"ece={ece} should have failed G2"


# ── G3: PR-AUC regression vs champion ────────────────────────

class TestG3PRaucRegression:

    def test_passes_no_champion(self):
        """No champion exists → first model, gate auto-passes."""
        candidate = _make_eval(pr_auc=0.80)
        passed, _ = check_promotion_gates(candidate, champion=None)
        assert passed

    def test_passes_acceptable_regression(self):
        champion  = _make_eval(pr_auc=0.85)
        candidate = _make_eval(pr_auc=0.84)   # -0.01 >= -0.02
        passed, failures = check_promotion_gates(candidate, champion=champion)
        assert passed

    def test_fails_excessive_regression(self):
        champion  = _make_eval(pr_auc=0.85)
        candidate = _make_eval(pr_auc=0.80)   # -0.05 < -0.02
        passed, failures = check_promotion_gates(candidate, champion=champion)
        assert not passed
        assert any("PR-AUC delta" in f or "regression" in f.lower() for f in failures)

    def test_passes_improvement_over_champion(self):
        champion  = _make_eval(pr_auc=0.80)
        candidate = _make_eval(pr_auc=0.85)   # improvement
        passed, _ = check_promotion_gates(candidate, champion=champion)
        assert passed


# ── G4: Recall regression vs champion ────────────────────────

class TestG4RecallRegression:

    def test_fails_large_recall_drop(self):
        champion  = _make_eval(recall=0.82)
        candidate = _make_eval(recall=0.75)   # -0.07 < -0.03
        passed, failures = check_promotion_gates(candidate, champion=champion)
        assert not passed
        assert any("Recall" in f or "recall" in f.lower() for f in failures)

    def test_passes_acceptable_recall_drop(self):
        champion  = _make_eval(recall=0.80)
        candidate = _make_eval(recall=0.78)   # -0.02 >= -0.03
        passed, _ = check_promotion_gates(candidate, champion=champion)
        assert passed


# ── Multi-gate failures ───────────────────────────────────────

class TestMultipleGateFailures:

    def test_all_gates_fail_simultaneously(self):
        """A truly terrible model should fail all gates at once."""
        candidate = _make_eval(pr_auc=0.50, ece=0.30, recall=0.10)
        champion  = _make_eval(pr_auc=0.85, recall=0.80)
        passed, failures = check_promotion_gates(candidate, champion=champion)
        assert not passed
        assert len(failures) >= 3, f"Expected ≥3 failures, got: {failures}"

    def test_gate_failure_list_is_non_empty_on_fail(self):
        candidate = _make_eval(pr_auc=0.60)
        passed, failures = check_promotion_gates(candidate, champion=None)
        assert not passed
        assert len(failures) > 0, "failures list must not be empty when gates fail"

    def test_gate_pass_has_empty_failure_list(self):
        candidate = _make_eval(pr_auc=0.82, ece=0.02, recall=0.76)
        passed, failures = check_promotion_gates(candidate, champion=None)
        assert passed
        assert failures == []


# ── Boundary conditions ───────────────────────────────────────

class TestGateBoundaryConditions:

    @pytest.mark.parametrize("pr_auc,expected_pass", [
        (0.75, True),    # exactly at floor
        (0.7499, False), # just below floor
        (0.7501, True),  # just above floor
    ])
    def test_g1_boundary(self, pr_auc: float, expected_pass: bool):
        candidate = _make_eval(pr_auc=pr_auc)
        passed, _ = check_promotion_gates(candidate, champion=None)
        assert passed == expected_pass, f"pr_auc={pr_auc}: expected passed={expected_pass}"

    @pytest.mark.parametrize("ece,expected_pass", [
        (0.05,  True),   # exactly at max
        (0.051, False),  # just over max
        (0.049, True),   # just under max
    ])
    def test_g2_boundary(self, ece: float, expected_pass: bool):
        candidate = _make_eval(pr_auc=0.80, ece=ece)
        passed, _ = check_promotion_gates(candidate, champion=None)
        assert passed == expected_pass, f"ece={ece}: expected passed={expected_pass}"
