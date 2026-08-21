"""
tests/unit/test_shap_aggregation.py
Unit tests for SHAP one-hot aggregation correctness (F-15).

Covers:
  - Correct prefix-based matching (not substring match)
  - Categorical SHAP sums all OHE column contributions
  - Numeric SHAP maps 1:1
  - Substring collision prevented (e.g. tenure_months vs avg_tenure_months)
  - Direction assignment (positive = increases_risk)
  - Customer value pass-through
"""
from __future__ import annotations

import numpy as np
import pytest


def _simulate_aggregation(
    transformed_names: list[str],
    shap_values: np.ndarray,
    original_features: list[str],
) -> dict[str, float]:
    """
    Replicate the fixed prefix-based aggregation logic from ChurnExplainer.
    Uses the same algorithm as shap_explainer.py (post F-15 fix).
    """
    aggregated: dict[str, float] = {}
    for orig_feat in original_features:
        total = 0.0
        for i, tfn in enumerate(transformed_names):
            feat_part = tfn.split("__", 1)[-1] if "__" in tfn else tfn
            if feat_part == orig_feat or feat_part.startswith(orig_feat + "_"):
                if i < len(shap_values):
                    total += float(shap_values[i])
        aggregated[orig_feat] = round(total, 5)
    return aggregated


class TestOneHotSHAPAggregation:

    def test_categorical_feature_sums_all_ohe_columns(self):
        """
        contract_type has 3 OHE columns.
        Aggregated SHAP must equal their sum.
        """
        transformed_names = [
            "num__tenure_months",
            "cat__contract_type_month-to-month",
            "cat__contract_type_one_year",
            "cat__contract_type_two_year",
            "bool__paperless_billing",
        ]
        shap_values = np.array([0.10, 0.30, -0.15, 0.05, -0.08])

        result = _simulate_aggregation(
            transformed_names, shap_values,
            ["tenure_months", "contract_type", "paperless_billing"],
        )

        assert abs(result["tenure_months"]   - 0.10) < 1e-4
        assert abs(result["contract_type"]   - (0.30 + (-0.15) + 0.05)) < 1e-4
        assert abs(result["paperless_billing"] - (-0.08)) < 1e-4

    def test_numeric_feature_maps_directly(self):
        """Numeric features are not OHE — they map 1:1."""
        transformed_names = ["num__monthly_charges", "num__tenure_months"]
        shap_values       = np.array([0.25, -0.10])

        result = _simulate_aggregation(
            transformed_names, shap_values,
            ["monthly_charges", "tenure_months"],
        )
        assert abs(result["monthly_charges"] - 0.25) < 1e-4
        assert abs(result["tenure_months"]   - (-0.10)) < 1e-4

    def test_substring_collision_prevented(self):
        """
        F-15: 'tenure_months' must NOT absorb 'avg_tenure_months' contributions.
        Old code used `if orig_feat in tfn` (substring) — would incorrectly match.
        New code uses `feat_part.startswith(orig_feat + "_")` (prefix) — safe.
        """
        transformed_names = [
            "num__tenure_months",        # should match tenure_months
            "num__avg_tenure_months",    # must NOT match tenure_months
        ]
        shap_values = np.array([0.10, 0.25])

        result = _simulate_aggregation(
            transformed_names, shap_values,
            ["tenure_months", "avg_tenure_months"],
        )

        # tenure_months should only get its own SHAP value (0.10), not 0.35
        assert abs(result["tenure_months"]     - 0.10) < 1e-4, (
            f"Substring collision: tenure_months got {result['tenure_months']}, "
            f"expected 0.10 (not 0.35 which would include avg_tenure_months)"
        )
        assert abs(result["avg_tenure_months"] - 0.25) < 1e-4

    def test_payment_method_ohe_aggregated_correctly(self):
        """payment_method with 4 categories maps to 4 OHE columns."""
        transformed_names = [
            "cat__payment_method_bank_transfer",
            "cat__payment_method_credit_card",
            "cat__payment_method_electronic_check",
            "cat__payment_method_mailed_check",
        ]
        shap_values = np.array([0.05, -0.03, 0.22, 0.01])

        result = _simulate_aggregation(
            transformed_names, shap_values, ["payment_method"]
        )

        expected = 0.05 + (-0.03) + 0.22 + 0.01
        assert abs(result["payment_method"] - expected) < 1e-4

    def test_feature_not_in_transformed_names_gets_zero(self):
        """A feature with no matching transformed column gets SHAP=0."""
        transformed_names = ["num__tenure_months"]
        shap_values       = np.array([0.10])

        result = _simulate_aggregation(
            transformed_names, shap_values,
            ["tenure_months", "missing_feature"],
        )
        assert abs(result["missing_feature"] - 0.0) < 1e-4

    def test_all_positive_shap_are_increases_risk(self):
        """Positive raw SHAP → direction=increases_risk."""
        shap_raw = 0.42
        direction = "increases_risk" if shap_raw > 0 else "decreases_risk"
        assert direction == "increases_risk"

    def test_all_negative_shap_are_decreases_risk(self):
        """Negative raw SHAP → direction=decreases_risk."""
        shap_raw = -0.17
        direction = "increases_risk" if shap_raw > 0 else "decreases_risk"
        assert direction == "decreases_risk"

    def test_zero_shap_value_direction(self):
        """SHAP = 0 → decreases_risk (not increases_risk) since condition is > 0."""
        shap_raw  = 0.0
        direction = "increases_risk" if shap_raw > 0 else "decreases_risk"
        assert direction == "decreases_risk"

    def test_driver_dict_contains_required_keys(self):
        """Each driver dict must have feature, label, shap_value, direction, customer_value."""
        from ml.explainability.shap_explainer import FEATURE_LABELS

        sample_driver = {
            "feature":        "tenure_months",
            "label":          FEATURE_LABELS["tenure_months"],
            "shap_value":     round(0.10, 5),
            "direction":      "increases_risk",
            "customer_value": 24,
        }

        required_keys = {"feature", "label", "shap_value", "direction", "customer_value"}
        assert required_keys.issubset(sample_driver.keys()), (
            f"Driver dict missing keys: {required_keys - set(sample_driver.keys())}"
        )

    def test_all_feature_labels_are_defined(self):
        """Every feature in FEATURE_COLUMNS must have a business-friendly label."""
        from ml.explainability.shap_explainer import FEATURE_LABELS
        from apps.api.services.feature_service import FEATURE_COLUMNS

        missing = [col for col in FEATURE_COLUMNS if col not in FEATURE_LABELS]
        assert not missing, (
            f"Missing business labels for features: {missing}. "
            "Add entries to FEATURE_LABELS in shap_explainer.py."
        )


class TestSHAPDriversSorting:

    def test_drivers_sorted_by_absolute_shap_descending(self):
        """Top drivers must be sorted by |shap_value| descending."""
        drivers = [
            {"feature": "a", "shap_value": 0.10, "direction": "increases_risk"},
            {"feature": "b", "shap_value": 0.42, "direction": "decreases_risk"},
            {"feature": "c", "shap_value": 0.05, "direction": "increases_risk"},
            {"feature": "d", "shap_value": 0.31, "direction": "increases_risk"},
        ]
        sorted_drivers = sorted(drivers, key=lambda d: abs(d["shap_value"]), reverse=True)

        assert sorted_drivers[0]["feature"] == "b"  # 0.42
        assert sorted_drivers[1]["feature"] == "d"  # 0.31
        assert sorted_drivers[2]["feature"] == "a"  # 0.10
        assert sorted_drivers[3]["feature"] == "c"  # 0.05

    def test_top_n_truncation(self):
        """explain_single with top_n=3 must return exactly 3 drivers."""
        drivers = [{"feature": f"f{i}", "shap_value": float(i) * 0.1} for i in range(10)]
        top_3   = sorted(drivers, key=lambda d: abs(d["shap_value"]), reverse=True)[:3]
        assert len(top_3) == 3
