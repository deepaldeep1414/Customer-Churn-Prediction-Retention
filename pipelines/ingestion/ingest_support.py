"""
pipelines/ingestion/ingest_support.py
Ingest support events from CSV → raw.support_raw → curated.support_events.
Full validation with quarantine routing and audit log.

Usage:
    python ingest_support.py --source-file data/support.csv --batch-id 2024-01-15-support-01
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
    validate_support_record,
    check_null_rates,
    check_duplicate_keys,
    ValidationFailure,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DATABASE_URL = os.getenv("DATABASE_URL")
BATCH_SIZE   = 500


async def _quarantine(conn, payload, failure, batch_id):
    await conn.execute(
        """
        INSERT INTO raw.quarantine
            (source_table, source_batch_id, source_payload,
             failure_category, failure_detail, failure_fields)
        VALUES ($1, $2, $3::jsonb, $4, $5, $6)
        """,
        "curated.support_events", batch_id,
        json.dumps(payload, default=str),
        failure.failure_category, failure.failure_detail,
        failure.failure_fields or [],
    )


async def ingest_support_csv(source_file: Path, batch_id: str, source_system: str = "csv") -> dict:
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=8)
    raw_inserted = curated_inserted = quarantined = 0

    with open(source_file, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    # Batch-level duplicate check on ticket external_id
    dupe_failures = check_duplicate_keys(all_rows, key_fields=["event_id"])
    for f in dupe_failures:
        logger.warning("Batch duplicate check [%s]: %s", f.failure_category, f.failure_detail)

    for batch_start in range(0, len(all_rows), BATCH_SIZE):
        batch = all_rows[batch_start : batch_start + BATCH_SIZE]

        async with pool.acquire() as conn:
            async with conn.transaction():
                for row in batch:
                    await conn.execute(
                        """
                        INSERT INTO raw.support_raw (source_system, payload, batch_id)
                        VALUES ($1, $2::jsonb, $3)
                        """,
                        source_system, json.dumps(row, default=str), batch_id,
                    )
                    raw_inserted += 1

                    result = validate_support_record(row)
                    if not result.is_valid:
                        for failure in result.failures:
                            await _quarantine(conn, row, failure, batch_id)
                        quarantined += 1
                        continue

                    try:
                        await conn.execute(
                            """
                            INSERT INTO curated.support_events
                                (customer_id, ticket_category, ticket_status,
                                 opened_at, resolved_at, csat_score)
                            VALUES (
                                (SELECT customer_id FROM curated.customers WHERE external_id = $1),
                                $2::curated.ticket_category,
                                $3::curated.ticket_status,
                                $4::TIMESTAMPTZ,
                                $5::TIMESTAMPTZ,
                                $6::SMALLINT
                            )
                            ON CONFLICT DO NOTHING
                            """,
                            row.get("customer_id"),
                            row.get("ticket_category", "general"),
                            row.get("ticket_status", "open"),
                            row.get("opened_at"),
                            row.get("resolved_at") or None,
                            int(row["csat_score"]) if row.get("csat_score") else None,
                        )
                        curated_inserted += 1
                    except Exception as exc:
                        logger.warning("Support insert failed for row %s: %s", row, exc)
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

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO raw.quality_audit_log
                (batch_id, source_table, total_records,
                 passed_records, quarantined_records, failure_summary)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            """,
            batch_id, "curated.support_events",
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
    logger.info("Support events ingestion complete: %s", summary)
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-file", required=True)
    p.add_argument("--batch-id",
                   default=f"support-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
    p.add_argument("--source-system", default="csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(ingest_support_csv(Path(args.source_file), args.batch_id, args.source_system))
