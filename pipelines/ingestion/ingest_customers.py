"""
pipelines/ingestion/ingest_customers.py
Ingest customer records from a CSV/JSONL source into raw.customers_raw,
then upsert into curated.customers with PII hashing.

Usage:
    python ingest_customers.py --source-file data/customers.csv --batch-id 2024-01-15-batch-01
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import asyncpg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DATABASE_URL = os.getenv("DATABASE_URL")
BATCH_SIZE   = 500


def _hash_pii(value: str | None) -> str | None:
    """SHA-256 hash of lowercase-stripped PII value."""
    if not value:
        return None
    return hashlib.sha256(value.strip().lower().encode()).hexdigest()


async def ingest_csv(source_file: Path, batch_id: str, source_system: str = "csv") -> None:
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=8)
    raw_inserted = curated_upserted = 0

    with open(source_file, newline="", encoding="utf-8") as f:
        reader   = csv.DictReader(f)
        batch    = []

        async def flush_batch(rows: list[dict]) -> None:
            nonlocal raw_inserted, curated_upserted
            async with pool.acquire() as conn:
                async with conn.transaction():
                    # ── Raw layer ──────────────────────────────
                    for row in rows:
                        await conn.execute(
                            """
                            INSERT INTO raw.customers_raw (source_system, payload, batch_id)
                            VALUES ($1, $2::jsonb, $3)
                            """,
                            source_system, json.dumps(row), batch_id,
                        )
                    raw_inserted += len(rows)

                    # ── Curated layer (PII-minimised upsert) ───
                    for row in rows:
                        email_hash = _hash_pii(row.get("email"))
                        phone_hash = _hash_pii(row.get("phone"))
                        if not email_hash:
                            logger.warning("Row missing email, skipping: %s", row.get("external_id"))
                            continue
                        await conn.execute(
                            """
                            INSERT INTO curated.customers
                                (external_id, email_hash, phone_hash, gender,
                                 senior_citizen, partner, dependents, signup_date,
                                 updated_at)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::DATE, now())
                            ON CONFLICT (external_id) DO UPDATE SET
                                email_hash    = EXCLUDED.email_hash,
                                phone_hash    = EXCLUDED.phone_hash,
                                senior_citizen = EXCLUDED.senior_citizen,
                                partner       = EXCLUDED.partner,
                                dependents    = EXCLUDED.dependents,
                                updated_at    = now()
                            """,
                            row.get("customer_id") or row.get("external_id"),
                            email_hash,
                            phone_hash,
                            row.get("gender", "unknown"),
                            row.get("senior_citizen", "0") in ("1", "true", "True"),
                            row.get("partner", "No").lower() == "yes",
                            row.get("dependents", "No").lower() == "yes",
                            row.get("signup_date") or row.get("join_date"),
                        )
                    curated_upserted += len(rows)

        for row in reader:
            batch.append(row)
            if len(batch) >= BATCH_SIZE:
                await flush_batch(batch)
                logger.info("Flushed batch | raw=%d curated=%d", raw_inserted, curated_upserted)
                batch = []

        if batch:
            await flush_batch(batch)

    await pool.close()
    logger.info(
        "Ingestion complete | raw_inserted=%d | curated_upserted=%d | batch_id=%s",
        raw_inserted, curated_upserted, batch_id,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-file", required=True)
    p.add_argument("--batch-id", default=f"batch-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
    p.add_argument("--source-system", default="csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(ingest_csv(Path(args.source_file), args.batch_id, args.source_system))
