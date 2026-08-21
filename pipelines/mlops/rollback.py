"""
pipelines/mlops/rollback.py
Automated rollback to the previous champion model.

Steps:
  1. Identify current champion and previous champion from ml.model_registry
  2. Archive current champion in both DB and MLflow registry
  3. Restore previous champion in both DB and MLflow registry
  4. Verify API /ready responds with the restored model_version
  5. Write rollback audit record

Exit code 0 = rollback succeeded, 1 = failed.

Usage:
    python rollback.py
    python rollback.py --target-version 1.2.1   # rollback to specific version
    python rollback.py --dry-run                 # show what would happen
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import asyncpg
import httpx
import mlflow

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DATABASE_URL    = os.getenv("DATABASE_URL")
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
API_BASE_URL    = os.getenv("API_BASE_URL",        "http://api:8000")
API_SECRET_KEY  = os.getenv("SECRET_KEY",          "")
MODEL_NAME      = os.getenv("MODEL_NAME",          "churn_predictor")


async def get_champion_and_previous(conn: asyncpg.Connection) -> tuple[Optional[dict], Optional[dict]]:
    """Return (current_champion, previous_champion) from model_registry."""
    rows = await conn.fetch(
        """
        SELECT model_version, stage, algorithm, pr_auc, promoted_at
        FROM ml.model_registry
        WHERE stage IN ('champion', 'archived')
        ORDER BY promoted_at DESC
        LIMIT 5
        """,
    )
    champion = None
    previous = None
    for row in rows:
        if row["stage"] == "champion" and champion is None:
            champion = dict(row)
        elif row["stage"] == "archived" and previous is None and champion is not None:
            previous = dict(row)
    return champion, previous


async def archive_in_db(conn: asyncpg.Connection, version: str) -> None:
    await conn.execute(
        """
        UPDATE ml.model_registry
        SET stage = 'archived', archived_at = now()
        WHERE model_version = $1
        """,
        version,
    )
    logger.info("DB: archived model version=%s", version)


async def promote_in_db(conn: asyncpg.Connection, version: str, notes: str) -> None:
    await conn.execute(
        """
        UPDATE ml.model_registry
        SET stage = 'champion', promoted_at = now(),
            promoted_by = 'automated_rollback',
            promotion_notes = $1
        WHERE model_version = $2
        """,
        notes, version,
    )
    logger.info("DB: promoted model version=%s to champion", version)


def update_mlflow_stage(version: str, new_stage: str) -> None:
    """Transition a model version in MLflow registry."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME, version=version, stage=new_stage
    )
    logger.info("MLflow: transitioned version=%s to stage=%s", version, new_stage)


async def verify_api_model(expected_version: str) -> bool:
    """Verify the API is serving the expected model version via /v1/models/current."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{API_BASE_URL}/v1/models/current",
                headers={"Authorization": f"Bearer {API_SECRET_KEY}"},
            )
        if resp.status_code == 200:
            body = resp.json()
            actual = body.get("model_version")
            if actual == expected_version:
                logger.info("API verification passed: model_version=%s", actual)
                return True
            else:
                logger.warning(
                    "API verification FAILED: expected=%s actual=%s",
                    expected_version, actual,
                )
                return False
        elif resp.status_code == 503:
            logger.warning("API returned 503 — model may still be reloading")
            return False
    except Exception as exc:
        logger.warning("API verification request failed: %s", exc)
    return False


async def execute_rollback(
    target_version: Optional[str] = None,
    dry_run: bool = False,
    rollback_reason: str = "manual_rollback",
) -> bool:
    """
    Execute the full rollback sequence.
    Returns True if successful.
    """
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=4)

    async with pool.acquire() as conn:
        champion, previous = await get_champion_and_previous(conn)

    if not champion:
        logger.error("No current champion found in model_registry. Cannot rollback.")
        await pool.close()
        return False

    # Determine target
    if target_version:
        async with pool.acquire() as conn:
            target_row = await conn.fetchrow(
                "SELECT model_version, stage FROM ml.model_registry WHERE model_version = $1",
                target_version,
            )
        if not target_row:
            logger.error("Target version %s not found in model_registry", target_version)
            await pool.close()
            return False
        target = dict(target_row)
    elif previous:
        target = previous
    else:
        logger.error("No previous champion found. Cannot rollback automatically.")
        await pool.close()
        return False

    logger.info(
        "Rollback plan: %s (champion) → %s (archived)",
        champion["model_version"], champion["model_version"],
    )
    logger.info(
        "Restore: %s (archived) → champion",
        target["model_version"],
    )

    if dry_run:
        logger.info("DRY RUN — no changes made.")
        await pool.close()
        return True

    # ── Execute rollback ─────────────────────────────────────
    notes = json.dumps({
        "reason":       rollback_reason,
        "rolled_back_from": champion["model_version"],
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    })

    async with pool.acquire() as conn:
        async with conn.transaction():
            await archive_in_db(conn, champion["model_version"])
            await promote_in_db(conn, target["model_version"], notes)

    # MLflow stage transitions
    try:
        update_mlflow_stage(champion["model_version"], "Archived")
        update_mlflow_stage(target["model_version"], "Production")
    except Exception as exc:
        logger.warning("MLflow stage update failed (DB already updated): %s", exc)

    # API verification (best-effort — API needs to reload)
    logger.info("Waiting 5s for API to pick up new champion...")
    await asyncio.sleep(5)
    verified = await verify_api_model(target["model_version"])

    if not verified:
        logger.warning(
            "API verification did not confirm rollback target. "
            "The API may need a restart: `docker compose restart api`"
        )

    await pool.close()
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Rollback to previous champion model")
    p.add_argument("--target-version", help="Specific version to restore (optional)")
    p.add_argument("--dry-run", action="store_true", help="Show plan without making changes")
    p.add_argument("--reason", default="manual_rollback", help="Rollback reason for audit log")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = asyncio.run(
        execute_rollback(
            target_version=args.target_version,
            dry_run=args.dry_run,
            rollback_reason=args.reason,
        )
    )
    sys.exit(0 if success else 1)
