"""
pipelines/ingestion/ingest_billing.py
Ingest billing records from CSV → raw.billing_raw → curated.billing.
Applies full data-quality validation and routes failures to raw.quarantine.

Usage:
    python ingest_billing.py --source-file data/billing.csv --batch-id 2024-01-15-billing-01
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import asyncpg

from ml.features.validators import (
    validate_billing_record,
    check_null_rates,
    check_duplicate_keys,
    ValidationFailure,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DATABASE_URL = os.getenv("DATABASE_URL")
BATCH_SIZE   = 500


async def _quarantine(
    conn: asyncpg.Connection,
    payload: dict,
    failure: ValidationFailure,
    batch_id: str,
) -> None:
    """Write a rejected record to raw.quarantine."""
    await conn.execute(
        """
        INSERT INTO raw.quarantine
            (source_table, source_batch_id, source_payload,
             failure_category, failure_detail, failure_fields)
        VALUES ($1, $2, $3::jsonb, $4, $5, $6)
        """,
        "curated.billing",
        batch_id,
        json.dumps(payload, default=str),
        failure.failure_category,
        failure.failure_detail,
        failure.failure_fields or [],
    )


async def ingest_billing_csv(
    source_file: Path,
    batch_id: str,
    source_system: str = "csv",
) -> dict:
    """
    Ingest billing records with full data-quality checks.

    Returns summary dict with counts.
    """
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=8)
    raw_inserted = curated_inserted = quarantined = 0

    with open(source_file, newline="", encoding="utf-8") as f:
        reader  = csv.DictReader(f)
        all_rows = list(reader)

    # ── Batch-level checks ───────────────────────────────────
    null_failures = check_null_rates(all_rows, record_type="billing")
    dupe_failures = check_duplicate_keys(
        all_rows,
        key_fields=["customer_id", "billing_period_start"],
    )

    if null_failures or dupe_failures:
        for f in null_failures + dupe_failures:
            logger.warning(
                "Batch-level validation failure [%s]: %s",
                f.failure_category, f.failure_detail,
            )

    # ── Row-level processing ─────────────────────────────────
    for batch_start in range(0, len(all_rows), BATCH_SIZE):
        batch = all_rows[batch_start : batch_start + BATCH_SIZE]

        async with pool.acquire() as conn:
            async with conn.transaction():
                for row in batch:
                    # Raw layer — always insert
                    await conn.execute(
                        """
                        INSERT INTO raw.billing_raw (source_system, payload, batch_id)
                        VALUES ($1, $2::jsonb, $3)
                        """,
                        source_system, json.dumps(row, default=str), batch_id,
                    )
                    raw_inserted += 1

                    # Validate
                    result = validate_billing_record(row)
                    if not result.is_valid:
                        for failure in result.failures:
                            await _quarantine(conn, row, failure, batch_id)
                        quarantined += 1
                        continue

                    # Curated upsert
                    try:
                        await conn.execute(
                            """
                            INSERT INTO curated.billing
                                (customer_id, billing_period_start, billing_period_end,
                                 monthly_charges, total_charges, payment_method,
                                 paperless_billing, payment_date, days_overdue)
                            VALUES (
                                (SELECT customer_id FROM curated.customers WHERE external_id = $1),
                                $2::DATE, $3::DATE, $4::NUMERIC, $5::NUMERIC,
                                $6::curated.payment_method, $7::BOOLEAN, $8::DATE, $9::INTEGER
                            )
                            ON CONFLICT DO NOTHING
                            """,
                            row.get("customer_id"),
                            row.get("billing_period_start"),
                            row.get("billing_period_end"),
                            row.get("monthly_charges"),
                            row.get("total_charges"),
                            row.get("payment_method", "electronic_check"),
                            row.get("paperless_billing", "False").lower() in ("true", "1", "yes"),
                            row.get("payment_date") or None,
                            int(row.get("days_overdue", 0)),
                        )
                        curated_inserted += 1
                    except Exception as exc:
                        logger.warning("Curated insert failed for row %s: %s", row, exc)
                        await _quarantine(
                            conn, row,
                            ValidationFailure(
                                failure_category="schema_mismatch",
                                failure_detail=f"DB insert error: {exc}",
                                source_payload=row,
                            ),
                            batch_id,
                        )
                        quarantined += 1

        logger.info(
            "Billing ingest progress | raw=%d curated=%d quarantined=%d",
            raw_inserted, curated_inserted, quarantined,
        )

    # ── Write audit log ──────────────────────────────────────
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO raw.quality_audit_log
                (batch_id, source_table, total_records,
                 passed_records, quarantined_records, failure_summary)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            """,
            batch_id, "curated.billing",
            len(all_rows), curated_inserted, quarantined,
            json.dumps({"quarantine_rate": quarantined / max(len(all_rows), 1)}),
        )

    await pool.close()
    summary = {
        "batch_id":        batch_id,
        "total":           len(all_rows),
        "raw_inserted":    raw_inserted,
        "curated_inserted": curated_inserted,
        "quarantined":     quarantined,
    }
    logger.info("Billing ingestion complete: %s", summary)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Ingest billing data")
    p.add_argument("--source-file", required=True)
    p.add_argument("--batch-id",
                   default=f"billing-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
    p.add_argument("--source-system", default="csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(ingest_billing_csv(Path(args.source_file), args.batch_id, args.source_system))
