"""
ml/explainability/shap_explainer.py
SHAP-based global and local explanations for the champion model.

Produces:
  - Global SHAP summary (feature importance bar chart + beeswarm)
  - Per-customer top-N drivers (stored in ml.predictions.top_drivers)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature display labels (business-friendly)
FEATURE_LABELS: dict[str, str] = {
    "tenure_months":            "Customer Tenure (months)",
    "monthly_charges":          "Current Monthly Charge ($)",
    "total_charges":            "Lifetime Spend ($)",
    "avg_monthly_charges_3m":   "Avg Monthly Charge (last 3 mo.)",
    "charge_change_pct_3m":     "Charge Change % (3 mo.)",
    "days_overdue_last":        "Days Overdue on Last Bill",
    "contract_type":            "Contract Type",
    "payment_method":           "Payment Method",
    "paperless_billing":        "Paperless Billing",
    "phone_service":            "Has Phone Service",
    "internet_service":         "Internet Service Type",
    "addon_count":              "Number of Add-On Services",
    "tickets_30d":              "Support Tickets (last 30 days)",
    "tickets_90d":              "Support Tickets (last 90 days)",
    "tickets_total":            "Total Support Tickets (all time)",
    "days_since_last_ticket":   "Days Since Last Support Contact",
    "pct_billing_tickets_90d":  "% Billing Tickets (last 90 days)",
    "pct_cancel_tickets_90d":   "% Cancellation-Intent Tickets (90 days)",
    "avg_csat_90d":             "Avg Satisfaction Score (last 90 days)",
    "charge_x_tenure":          "Charge × Tenure (interaction)",
    "overdue_x_cancel_intent":  "Overdue Bill × Cancellation Intent (interaction)",
}


class ChurnExplainer:
    """
    Wraps shap.TreeExplainer (or LinearExplainer) for the champion pipeline.
    Handles both sklearn Pipeline wrapping and direct model access.
    """

    def __init__(self, pipeline: Any, algorithm: str, X_background: pd.DataFrame) -> None:
        import shap

        self.algorithm = algorithm
        self.feature_names = list(X_background.columns)

        # Extract the raw model from sklearn Pipeline if needed
        if hasattr(pipeline, "named_steps"):
            # Get the preprocessed background for explainer
            preprocessor = pipeline.named_steps.get("preprocessor")
            model_step   = pipeline.named_steps.get("model")

            # For calibrated models, drill into base estimator
            if hasattr(model_step, "estimator"):
                model_step = model_step.estimator

            X_bg_transformed = preprocessor.transform(X_background)
            self._preprocessor = preprocessor

            try:
                self._explainer = shap.TreeExplainer(model_step)
                self._transformed_input = True
            except Exception:
                self._explainer = shap.LinearExplainer(model_step, X_bg_transformed)
                self._transformed_input = True
        else:
            self._preprocessor = None
            self._transformed_input = False
            self._explainer = shap.TreeExplainer(pipeline)

        logger.info("SHAP explainer initialised | algorithm=%s", algorithm)

    def explain_single(
        self,
        X_row: pd.DataFrame,
        customer_raw_values: dict,
        top_n: int = 5,
    ) -> list[dict]:
        """
        Compute SHAP values for a single customer row.

        Returns a list of dicts (sorted by |shap_value| desc), truncated to top_n:
          {feature, label, shap_value, direction, customer_value}
        """
        if self._transformed_input and self._preprocessor is not None:
            X_input = self._preprocessor.transform(X_row)
        else:
            X_input = X_row.values

        shap_vals = self._explainer(X_input)
        # shap_vals.values shape: (1, n_features)
        values    = shap_vals.values[0]

        # Map back to original feature names (pre-transform)
        # For one-hot encoded features, aggregate SHAP per original feature
        drivers = self._aggregate_to_original_features(values, X_row, customer_raw_values)

        sorted_drivers = sorted(drivers, key=lambda d: abs(d["shap_value"]), reverse=True)
        return sorted_drivers[:top_n]

    def _aggregate_to_original_features(
        self,
        shap_values: np.ndarray,
        X_original: pd.DataFrame,
        raw_values: dict,
    ) -> list[dict]:
        """
        If a preprocessor exists, aggregate one-hot encoded SHAP values
        back to original categorical features by summing their SHAP contributions.
        """
        if self._preprocessor is None:
            # Direct mapping
            return [
                {
                    "feature":        col,
                    "label":          FEATURE_LABELS.get(col, col),
                    "shap_value":     round(float(sv), 5),
                    "direction":      "increases_risk" if sv > 0 else "decreases_risk",
                    "customer_value": raw_values.get(col),
                }
                for col, sv in zip(self.feature_names, shap_values)
            ]

        # Build feature name → shap_value mapping from transformed space
        try:
            transformed_feature_names = list(self._preprocessor.get_feature_names_out())
        except Exception:
            transformed_feature_names = [f"f{i}" for i in range(len(shap_values))]

        # Aggregate: for each original feature, sum SHAP values of its OHE columns.
        #
        # ColumnTransformer produces names like:
        #   "num__tenure_months"             → numeric passthrough
        #   "cat__contract_type_one_year"    → OHE column for contract_type
        #   "bool__paperless_billing"        → boolean passthrough
        #
        # Matching rule:
        #   - Exact match after prefix strip:  feat_part == orig_feat
        #   - OHE child column:                feat_part.startswith(orig_feat + "_")
        #
        # This prevents "tenure_months" from matching "avg_tenure_months"
        # because startswith("tenure_months_") won't match "avg_tenure_months".
        aggregated: dict[str, float] = {}
        for orig_feat in self.feature_names:
            total = 0.0
            for i, tfn in enumerate(transformed_feature_names):
                # Strip the ColumnTransformer prefix (e.g. "num__", "cat__", "bool__")
                feat_part = tfn.split("__", 1)[-1] if "__" in tfn else tfn
                # Match exact name OR OHE child column (e.g. "contract_type_one_year")
                if feat_part == orig_feat or feat_part.startswith(orig_feat + "_"):
                    if i < len(shap_values):
                        total += float(shap_values[i])
            aggregated[orig_feat] = total

        return [
            {
                "feature":        feat,
                "label":          FEATURE_LABELS.get(feat, feat),
                "shap_value":     round(sv, 5),
                "direction":      "increases_risk" if sv > 0 else "decreases_risk",
                "customer_value": raw_values.get(feat),
            }
            for feat, sv in aggregated.items()
        ]


    def global_feature_importance(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Compute mean |SHAP| across the test set for each feature.
        Returns a DataFrame sorted by importance desc.
        """
        if self._transformed_input and self._preprocessor is not None:
            X_input = self._preprocessor.transform(X_test)
        else:
            X_input = X_test.values

        shap_vals = self._explainer(X_input)
        mean_abs  = np.abs(shap_vals.values).mean(axis=0)

        try:
            feature_names = self._preprocessor.get_feature_names_out()
        except Exception:
            feature_names = self.feature_names

        importance_df = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        return importance_df

    def save_summary_plot(self, X_test: pd.DataFrame, output_path: Path) -> None:
        """Save SHAP beeswarm summary plot to disk for MLflow logging."""
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if self._transformed_input and self._preprocessor is not None:
            X_input = self._preprocessor.transform(X_test)
        else:
            X_input = X_test.values

        shap_vals = self._explainer(X_input)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, X_input, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary plot saved to %s", output_path)
