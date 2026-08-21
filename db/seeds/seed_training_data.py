import asyncio
import os
import random
from datetime import date, timedelta
import asyncpg

DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql://churn_admin:PostgresAdminPassword2026!@localhost:5432/churn_db"

async def seed():
    conn = await asyncpg.connect(DATABASE_URL)
    # Fetch customer IDs
    custs = await conn.fetch("SELECT customer_id FROM curated.customers")
    if not custs:
        print("No customers found in curated.customers. Run sample_data seed first.")
        await conn.close()
        return

    dates = [
        date(2023, 1, 15), date(2023, 3, 30), date(2023, 5, 15),
        date(2023, 6, 30), date(2023, 8, 15), date(2023, 9, 30),
        date(2023, 11, 15), date(2023, 12, 31), date(2024, 1, 1)
    ]

    inserted = 0
    for d in dates:
        for i, c in enumerate(custs):
            cid = c["customer_id"]
            # 25% churn rate
            churned = (i % 4 == 0)
            churn_date = (d + timedelta(days=15)) if churned else None

            # Insert churn label with churn_date if churned
            await conn.execute("""
                INSERT INTO curated.churn_labels (
                    customer_id, observation_date, prediction_horizon, churned, churn_date
                ) VALUES ($1, $2, 30, $3, $4)
                ON CONFLICT DO NOTHING
            """, cid, d, churned, churn_date)

            # Insert feature snapshot
            tenure = random.randint(1, 48)
            monthly = round(random.uniform(25.0, 110.0), 2)
            total = round(monthly * tenure, 2)
            contract = random.choice(['month-to-month', 'one_year', 'two_year'])
            payment = random.choice(['electronic_check', 'mailed_check', 'bank_transfer', 'credit_card'])

            await conn.execute("""
                INSERT INTO ml.feature_snapshots (
                    customer_id, observation_date, prediction_horizon_days, feature_set_version,
                    tenure_months, monthly_charges, total_charges, contract_type, payment_method, paperless_billing,
                    phone_service, internet_service, addon_count, tickets_30d, tickets_90d, tickets_total
                ) VALUES ($1, $2, 30, 'v1.0.0', $3, $4, $5, $6, $7, true, true, 'fiber_optic', 3, 1, 2, 5)
                ON CONFLICT DO NOTHING
            """, cid, d, tenure, monthly, total, contract, payment)
            inserted += 1

    await conn.close()
    print(f"Successfully seeded {inserted} multi-date training snapshots!")

if __name__ == "__main__":
    asyncio.run(seed())
