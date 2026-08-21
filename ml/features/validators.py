"""
ml/features/validators.py
Data-quality validation for ingested records and feature snapshots.

Validation categories (matching raw.quarantine.failure_category):
  - schema_mismatch    : Required field missing or wrong type
  - impossible_value   : Value outside physically possible range
  - excessive_nulls    : Critical field null rate > threshold
  - duplicate_key      : Duplicate natural key within a batch
  - leakage_risk       : Feature value implies future information

All failures are routed to raw.quarantine — never silently dropped.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)

# ── Null-rate thresholds per column ─────────────────────────
NULL_RATE_THRESHOLDS: dict[str, float] = {
    "monthly_charges":   0.05,   # > 5% nulls → quarantine
    "total_charges":     0.05,
    "contract_type":     0.02,
    "tenure_months":     0.01,
    "payment_method":    0.10,
    "phone_service":     0.02,
}

# ── Sanity bounds for numeric fields ─────────────────────────
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "monthly_charges":          (0.0,   10_000.0),
    "total_charges":            (0.0,  500_000.0),
    "tenure_months":            (0,        600),    # 50 years max
    "days_overdue_last":        (0,        366),
    "tickets_30d":              (0,      1_000),
    "tickets_90d":              (0,      3_000),
    "tickets_total":            (0,    100_000),
    "addon_count":              (0,          6),
    "pct_billing_tickets_90d":  (0.0,     1.0),
    "pct_cancel_tickets_90d":   (0.0,     1.0),
    "avg_csat_90d":             (1.0,     5.0),
    "churn_probability":        (0.0,     1.0),
}

# ── Required fields per record type ──────────────────────────
REQUIRED_FIELDS: dict[str, list[str]] = {
    "customer": ["external_id", "email", "signup_date"],
    "billing":  ["customer_id", "billing_period_start", "billing_period_end",
                 "monthly_charges", "total_charges", "payment_method"],
    "support":  ["customer_id", "ticket_category", "opened_at"],
    "feature_snapshot": [
        "customer_id", "observation_date", "tenure_months",
        "monthly_charges", "total_charges", "contract_type", "payment_method",
    ],
}


@dataclass
class ValidationFailure:
    """A single validation failure for one record."""
    failure_category: str
    failure_detail:   str
    failure_fields:   list[str] = field(default_factory=list)
    source_payload:   dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validating one record."""
    record:    dict
    is_valid:  bool
    failures:  list[ValidationFailure] = field(default_factory=list)


def validate_customer_record(record: dict) -> ValidationResult:
    """Validate a raw customer record before curated upsert."""
    failures: list[ValidationFailure] = []

    # ── Schema: required fields ──────────────────────────────
    missing = [f for f in REQUIRED_FIELDS["customer"] if not record.get(f)]
    if missing:
        failures.append(ValidationFailure(
            failure_category="schema_mismatch",
            failure_detail=f"Required fields missing: {missing}",
            failure_fields=missing,
            source_payload=record,
        ))

    # ── Impossible values ────────────────────────────────────
    signup = record.get("signup_date")
    if signup:
        try:
            signup_dt = date.fromisoformat(str(signup))
            if signup_dt > date.today():
                failures.append(ValidationFailure(
                    failure_category="impossible_value",
                    failure_detail=f"signup_date={signup} is in the future",
                    failure_fields=["signup_date"],
                    source_payload=record,
                ))
            if signup_dt.year < 1990:
                failures.append(ValidationFailure(
                    failure_category="impossible_value",
                    failure_detail=f"signup_date={signup} is implausibly old (< 1990)",
                    failure_fields=["signup_date"],
                    source_payload=record,
                ))
        except (ValueError, TypeError):
            failures.append(ValidationFailure(
                failure_category="schema_mismatch",
                failure_detail=f"signup_date={signup!r} is not a valid date",
                failure_fields=["signup_date"],
                source_payload=record,
            ))

    # ── Email format (basic) ─────────────────────────────────
    email = record.get("email", "")
    if email and "@" not in email:
        failures.append(ValidationFailure(
            failure_category="impossible_value",
            failure_detail=f"email={email!r} does not contain @",
            failure_fields=["email"],
            source_payload=record,
        ))

    return ValidationResult(
        record=record,
        is_valid=len(failures) == 0,
        failures=failures,
    )


def validate_billing_record(record: dict) -> ValidationResult:
    """Validate a raw billing record."""
    failures: list[ValidationFailure] = []

    # ── Schema ───────────────────────────────────────────────
    missing = [f for f in REQUIRED_FIELDS["billing"] if record.get(f) is None]
    if missing:
        failures.append(ValidationFailure(
            failure_category="schema_mismatch",
            failure_detail=f"Required fields missing: {missing}",
            failure_fields=missing,
            source_payload=record,
        ))

    # ── Numeric bounds ───────────────────────────────────────
    for field_name in ["monthly_charges", "total_charges"]:
        val = record.get(field_name)
        if val is not None:
            try:
                fval = float(val)
                lo, hi = NUMERIC_BOUNDS[field_name]
                if not (lo <= fval <= hi):
                    failures.append(ValidationFailure(
                        failure_category="impossible_value",
                        failure_detail=f"{field_name}={fval} out of bounds [{lo}, {hi}]",
                        failure_fields=[field_name],
                        source_payload=record,
                    ))
            except (TypeError, ValueError):
                failures.append(ValidationFailure(
                    failure_category="schema_mismatch",
                    failure_detail=f"{field_name}={val!r} is not numeric",
                    failure_fields=[field_name],
                    source_payload=record,
                ))

    # ── total_charges >= monthly_charges ─────────────────────
    mc = _safe_float(record.get("monthly_charges"))
    tc = _safe_float(record.get("total_charges"))
    if mc is not None and tc is not None and tc < mc:
        failures.append(ValidationFailure(
            failure_category="impossible_value",
            failure_detail=f"total_charges={tc} < monthly_charges={mc}",
            failure_fields=["total_charges", "monthly_charges"],
            source_payload=record,
        ))

    # ── Billing period sanity ────────────────────────────────
    ps = record.get("billing_period_start")
    pe = record.get("billing_period_end")
    if ps and pe:
        try:
            ps_dt = date.fromisoformat(str(ps))
            pe_dt = date.fromisoformat(str(pe))
            if pe_dt <= ps_dt:
                failures.append(ValidationFailure(
                    failure_category="impossible_value",
                    failure_detail=f"billing_period_end={pe} must be after billing_period_start={ps}",
                    failure_fields=["billing_period_start", "billing_period_end"],
                    source_payload=record,
                ))
            if pe_dt > date.today():
                failures.append(ValidationFailure(
                    failure_category="leakage_risk",
                    failure_detail=f"billing_period_end={pe} is in the future — potential future data",
                    failure_fields=["billing_period_end"],
                    source_payload=record,
                ))
        except (ValueError, TypeError):
            pass

    return ValidationResult(record=record, is_valid=len(failures) == 0, failures=failures)


def validate_support_record(record: dict) -> ValidationResult:
    """Validate a raw support event record."""
    failures: list[ValidationFailure] = []

    missing = [f for f in REQUIRED_FIELDS["support"] if not record.get(f)]
    if missing:
        failures.append(ValidationFailure(
            failure_category="schema_mismatch",
            failure_detail=f"Required fields missing: {missing}",
            failure_fields=missing,
            source_payload=record,
        ))

    valid_categories = {"billing", "technical", "general", "cancellation_intent"}
    cat = record.get("ticket_category", "")
    if cat and cat not in valid_categories:
        failures.append(ValidationFailure(
            failure_category="impossible_value",
            failure_detail=f"ticket_category={cat!r} not in {valid_categories}",
            failure_fields=["ticket_category"],
            source_payload=record,
        ))

    csat = record.get("csat_score")
    if csat is not None:
        try:
            if not (1 <= int(csat) <= 5):
                failures.append(ValidationFailure(
                    failure_category="impossible_value",
                    failure_detail=f"csat_score={csat} out of [1,5]",
                    failure_fields=["csat_score"],
                    source_payload=record,
                ))
        except (TypeError, ValueError):
            pass

    return ValidationResult(record=record, is_valid=len(failures) == 0, failures=failures)


def validate_feature_snapshot(features: dict, observation_date: date) -> ValidationResult:
    """
    Validate a fully-computed feature dict before insertion into ml.feature_snapshots.
    Includes leakage checks and numeric bounds.
    """
    failures: list[ValidationFailure] = []

    # ── Required fields ──────────────────────────────────────
    missing = [f for f in REQUIRED_FIELDS["feature_snapshot"] if features.get(f) is None]
    if missing:
        failures.append(ValidationFailure(
            failure_category="schema_mismatch",
            failure_detail=f"Required feature fields missing: {missing}",
            failure_fields=missing,
            source_payload=features,
        ))

    # ── Numeric bounds ───────────────────────────────────────
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        val = features.get(col)
        if val is None:
            continue
        try:
            fval = float(val)
            if not (lo <= fval <= hi):
                failures.append(ValidationFailure(
                    failure_category="impossible_value",
                    failure_detail=f"{col}={fval} out of bounds [{lo}, {hi}]",
                    failure_fields=[col],
                    source_payload=features,
                ))
        except (TypeError, ValueError):
            pass

    # ── Leakage check: no label columns ─────────────────────
    forbidden = {"churned", "churn_date", "label_id", "prediction_horizon",
                 "labeling_method", "label_version"}
    overlap = forbidden & set(features.keys())
    if overlap:
        failures.append(ValidationFailure(
            failure_category="leakage_risk",
            failure_detail=f"Label columns found in feature dict: {overlap}",
            failure_fields=list(overlap),
            source_payload=features,
        ))

    # ── Monotonicity: 30d ≤ 90d ≤ total ─────────────────────
    t30  = features.get("tickets_30d",  0) or 0
    t90  = features.get("tickets_90d",  0) or 0
    ttot = features.get("tickets_total", 0) or 0
    if t30 > t90:
        failures.append(ValidationFailure(
            failure_category="impossible_value",
            failure_detail=f"tickets_30d={t30} > tickets_90d={t90} (monotonicity violation)",
            failure_fields=["tickets_30d", "tickets_90d"],
            source_payload=features,
        ))
    if t90 > ttot:
        failures.append(ValidationFailure(
            failure_category="impossible_value",
            failure_detail=f"tickets_90d={t90} > tickets_total={ttot}",
            failure_fields=["tickets_90d", "tickets_total"],
            source_payload=features,
        ))

    return ValidationResult(record=features, is_valid=len(failures) == 0, failures=failures)


def check_null_rates(
    records: list[dict],
    record_type: str,
) -> list[ValidationFailure]:
    """
    Check null rates across a batch of records.
    Returns failures for columns exceeding NULL_RATE_THRESHOLDS.
    """
    if not records:
        return []
    failures = []
    n = len(records)
    for col, threshold in NULL_RATE_THRESHOLDS.items():
        null_count = sum(1 for r in records if r.get(col) is None)
        rate = null_count / n
        if rate > threshold:
            failures.append(ValidationFailure(
                failure_category="excessive_nulls",
                failure_detail=(
                    f"Column '{col}' has null rate {rate:.2%} > threshold {threshold:.2%} "
                    f"({null_count}/{n} records)"
                ),
                failure_fields=[col],
                source_payload={"null_count": null_count, "total": n, "null_rate": rate},
            ))
    return failures


def check_duplicate_keys(records: list[dict], key_fields: list[str]) -> list[ValidationFailure]:
    """Detect duplicate natural keys within a batch."""
    seen: set[tuple] = set()
    dupes: list[tuple] = []
    for r in records:
        key = tuple(r.get(f) for f in key_fields)
        if key in seen:
            dupes.append(key)
        seen.add(key)
    if dupes:
        return [ValidationFailure(
            failure_category="duplicate_key",
            failure_detail=f"Duplicate keys in batch: {dupes[:5]} (showing up to 5)",
            failure_fields=key_fields,
            source_payload={"duplicate_count": len(dupes)},
        )]
    return []


def _safe_float(val: Any) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None
