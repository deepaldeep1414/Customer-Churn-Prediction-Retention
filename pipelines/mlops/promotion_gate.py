"""
pipelines/mlops/promotion_gate.py
Candidate-vs-champion comparison and automated promotion gate enforcement.

Gates checked (all must pass for promotion):
  G1. Absolute PR-AUC floor          : candidate val_pr_auc >= 0.75
  G2. Calibration gate               : candidate val_ece   <= 0.05
  G3. PR-AUC no-regression vs champion : delta >= -0.02
  G4. Recall no-regression vs champion  : delta >= -0.03
  G5. Data-quality gate              : feature null rates < 5% on key columns
  G6. Reproducibility check          : train/val/test dataset hashes logged
  G7. Leakage audit                  : ml.validate_snapshot_leakage() returns 0 rows

Human sign-off (G8) is enforced via GitHub Actions environment gate, not here.

Usage:
    python promotion_gate.py \
        --candidate-run-id <mlflow_run_id> \
        --candidate-version 1.3.2 \
        --feature-version v1.0.0 \
        --observation-date 2024-01-01
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import asyncpg
import mlflow

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DATABASE_URL = os.getenv("DATABASE_URL")
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "churn_predictor")

# Gate thresholds
GATE_PR_AUC_MIN      = float(os.getenv("GATE_PR_AUC_MIN",    "0.75"))
GATE_ECE_MAX         = float(os.getenv("GATE_ECE_MAX",       "0.05"))
GATE_PR_AUC_DELTA    = float(os.getenv("GATE_PR_AUC_DELTA",  "-0.02"))
GATE_RECALL_DELTA    = float(os.getenv("GATE_RECALL_DELTA",  "-0.03"))
GATE_NULL_RATE_MAX   = float(os.getenv("GATE_NULL_RATE_MAX", "0.05"))


@dataclass
class GateResult:
    gate_id:  str
    name:     str
    passed:   bool
    value:    float | str | None = None
    threshold: float | str | None = None
    detail:   str = ""


@dataclass
class PromotionReport:
    candidate_run_id:     str
    candidate_version:    str
    champion_version:     Optional[str]
    gates:                list[GateResult] = field(default_factory=list)
    all_passed:           bool = False
    failure_summary:      list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidate_run_id":   self.candidate_run_id,
            "candidate_version":  self.candidate_version,
            "champion_version":   self.champion_version,
            "all_passed":         self.all_passed,
            "failure_summary":    self.failure_summary,
            "gates": [
                {
                    "gate_id":   g.gate_id,
                    "name":      g.name,
                    "passed":    g.passed,
                    "value":     g.value,
                    "threshold": g.threshold,
                    "detail":    g.detail,
                }
                for g in self.gates
            ],
        }


async def run_promotion_gates(
    candidate_run_id: str,
    candidate_version: str,
    feature_version: str,
    observation_date: date,
) -> PromotionReport:
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    run     = client.get_run(candidate_run_id)
    metrics = run.data.metrics
    params  = run.data.params

    # Get current champion metrics from DB
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=4)
    async with pool.acquire() as conn:
        champion = await conn.fetchrow(
            "SELECT model_version, pr_auc, recall_at_threshold "
            "FROM ml.model_registry WHERE stage='champion' LIMIT 1"
        )

    champion_version = champion["model_version"] if champion else None
    report = PromotionReport(
        candidate_run_id=candidate_run_id,
        candidate_version=candidate_version,
        champion_version=champion_version,
    )

    # ── G1: Absolute PR-AUC floor ────────────────────────────
    val_pr_auc = metrics.get("val_thresh_pr_auc") or metrics.get("val_pr_auc", 0.0)
    g1 = GateResult(
        gate_id="G1", name="Absolute PR-AUC floor",
        passed=val_pr_auc >= GATE_PR_AUC_MIN,
        value=round(val_pr_auc, 5),
        threshold=GATE_PR_AUC_MIN,
        detail=f"val_pr_auc={val_pr_auc:.4f}",
    )
    report.gates.append(g1)

    # ── G2: ECE calibration ───────────────────────────────────
    val_ece = metrics.get("val_cal_ece") or metrics.get("val_ece")
    if val_ece is not None:
        g2 = GateResult(
            gate_id="G2", name="Calibration (ECE) gate",
            passed=float(val_ece) <= GATE_ECE_MAX,
            value=round(float(val_ece), 5),
            threshold=GATE_ECE_MAX,
            detail=f"val_ece={val_ece:.4f}",
        )
    else:
        g2 = GateResult(
            gate_id="G2", name="Calibration (ECE) gate",
            passed=False,
            detail="val_ece not logged in run — calibration step may have been skipped",
        )
    report.gates.append(g2)

    # ── G3: PR-AUC no-regression vs champion ─────────────────
    if champion and champion["pr_auc"]:
        champ_pr = float(champion["pr_auc"])
        delta_pr = val_pr_auc - champ_pr
        g3 = GateResult(
            gate_id="G3", name="PR-AUC regression vs champion",
            passed=delta_pr >= GATE_PR_AUC_DELTA,
            value=round(delta_pr, 5),
            threshold=GATE_PR_AUC_DELTA,
            detail=f"candidate={val_pr_auc:.4f} champion={champ_pr:.4f} delta={delta_pr:+.4f}",
        )
    else:
        g3 = GateResult(
            gate_id="G3", name="PR-AUC regression vs champion",
            passed=True,
            detail="No champion exists — first model, gate auto-passed",
        )
    report.gates.append(g3)

    # ── G4: Recall regression vs champion ────────────────────
    val_recall = metrics.get("val_thresh_recall") or metrics.get("val_recall", 0.0)
    if champion and champion["recall_at_threshold"]:
        champ_recall = float(champion["recall_at_threshold"])
        delta_recall = val_recall - champ_recall
        g4 = GateResult(
            gate_id="G4", name="Recall regression vs champion",
            passed=delta_recall >= GATE_RECALL_DELTA,
            value=round(delta_recall, 5),
            threshold=GATE_RECALL_DELTA,
            detail=f"candidate={val_recall:.4f} champion={champ_recall:.4f} delta={delta_recall:+.4f}",
        )
    else:
        g4 = GateResult(
            gate_id="G4", name="Recall regression vs champion",
            passed=True,
            detail="No champion recall baseline — gate auto-passed",
        )
    report.gates.append(g4)

    # ── G5: Data-quality (null rates) ────────────────────────
    async with pool.acquire() as conn:
        null_check = await conn.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE monthly_charges IS NULL)::FLOAT / NULLIF(COUNT(*),0) AS null_monthly,
                COUNT(*) FILTER (WHERE tenure_months IS NULL)::FLOAT  / NULLIF(COUNT(*),0) AS null_tenure,
                COUNT(*) FILTER (WHERE contract_type IS NULL)::FLOAT  / NULLIF(COUNT(*),0) AS null_contract,
                COUNT(*)                                                                    AS total
            FROM ml.feature_snapshots
            WHERE observation_date    = $1
              AND feature_set_version = $2
            """,
            observation_date, feature_version,
        )
    bad_cols = {}
    for col in ["null_monthly", "null_tenure", "null_contract"]:
        rate = float(null_check[col] or 0)
        if rate > GATE_NULL_RATE_MAX:
            bad_cols[col.replace("null_", "")] = round(rate, 4)

    g5 = GateResult(
        gate_id="G5", name="Feature null-rate gate",
        passed=len(bad_cols) == 0,
        threshold=GATE_NULL_RATE_MAX,
        detail=f"Bad cols: {bad_cols}" if bad_cols else "All null rates within threshold",
        value=max(bad_cols.values()) if bad_cols else 0.0,
    )
    report.gates.append(g5)

    # ── G6: Reproducibility (dataset hashes logged) ───────────
    has_hashes = all(
        params.get(h) for h in ["train_dataset_hash", "val_dataset_hash", "git_commit"]
    )
    g6 = GateResult(
        gate_id="G6", name="Reproducibility (dataset hashes + git commit)",
        passed=has_hashes,
        detail="train/val hashes and git_commit found in run params" if has_hashes
               else "Missing: train_dataset_hash, val_dataset_hash, or git_commit",
    )
    report.gates.append(g6)

    # ── G7: Leakage audit (DB stored procedure) ───────────────
    async with pool.acquire() as conn:
        leaking = await conn.fetch(
            "SELECT * FROM ml.validate_snapshot_leakage($1, $2) LIMIT 10",
            feature_version, observation_date,
        )
    g7 = GateResult(
        gate_id="G7", name="Leakage audit (ml.validate_snapshot_leakage)",
        passed=len(leaking) == 0,
        value=len(leaking),
        threshold=0,
        detail=f"Leaking snapshots: {len(leaking)}" if leaking
               else "No leakage detected",
    )
    report.gates.append(g7)

    await pool.close()

    # ── Final decision ────────────────────────────────────────
    failures = [g for g in report.gates if not g.passed]
    report.all_passed     = len(failures) == 0
    report.failure_summary = [f"[{g.gate_id}] {g.name}: {g.detail}" for g in failures]

    icon = "✅" if report.all_passed else "❌"
    logger.info(
        "%s Promotion gate result for version=%s: %s",
        icon, candidate_version,
        "ALL PASSED" if report.all_passed else f"{len(failures)} GATE(S) FAILED",
    )
    for g in report.gates:
        status = "✅" if g.passed else "❌"
        logger.info("  %s [%s] %s: %s", status, g.gate_id, g.name, g.detail)

    await persist_gate_result(report)
    return report


async def persist_gate_result(report: PromotionReport) -> None:
    """Write promotion gate result to ml.model_registry and update champion stage."""
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=2)
    async with pool.acquire() as conn:
        if report.all_passed:
            # 1. Archive existing champion in DB
            await conn.execute(
                "UPDATE ml.model_registry SET stage = 'archived', archived_at = now() WHERE stage = 'champion'"
            )
            # 2. Promote candidate to champion
            await conn.execute(
                """
                UPDATE ml.model_registry
                SET stage = 'champion', promoted_at = now(), promotion_notes = $1
                WHERE model_version = $2
                """,
                json.dumps({"gates": "ALL PASSED"}),
                report.candidate_version,
            )
            logger.info("Promoted %s to stage 'champion' in ml.model_registry", report.candidate_version)

            # 3. Register and transition stage in MLflow Model Registry
            try:
                mlflow.set_tracking_uri(MLFLOW_URI)
                client = mlflow.tracking.MlflowClient()
                try:
                    client.create_registered_model(MODEL_NAME)
                except Exception:
                    pass
                model_src = f"runs:/{report.candidate_run_id}/model"
                mv = client.create_model_version(MODEL_NAME, model_src, report.candidate_run_id)
                client.transition_model_version_stage(
                    name=MODEL_NAME, version=mv.version, stage="Production", archive_existing_versions=True
                )
                logger.info("Transitioned MLflow model '%s' v%s to 'Production'", MODEL_NAME, mv.version)
            except Exception as exc:
                logger.warning("Failed to transition MLflow model stage (non-fatal): %s", exc)
        else:
            await conn.execute(
                """
                UPDATE ml.model_registry
                SET promotion_notes = $1
                WHERE model_version = $2
                """,
                json.dumps({"gates": [g.detail for g in report.gates]}),
                report.candidate_version,
            )
    await pool.close()


def parse_args():
    p = argparse.ArgumentParser(description="Run promotion gates for candidate model")
    p.add_argument("--candidate-run-id",  required=True)
    p.add_argument("--candidate-version", required=True)
    p.add_argument("--feature-version",   default="v1.0.0")
    p.add_argument("--observation-date",  required=True, help="YYYY-MM-DD")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = asyncio.run(
        run_promotion_gates(
            candidate_run_id=args.candidate_run_id,
            candidate_version=args.candidate_version,
            feature_version=args.feature_version,
            observation_date=date.fromisoformat(args.observation_date),
        )
    )
    print(json.dumps(report.to_dict(), indent=2))
    sys.exit(0 if report.all_passed else 1)
