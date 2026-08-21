-- ============================================================
-- ml/features/feature_query.sql
-- Point-in-time safe feature extraction query.
--
-- Parameters:
--   $1 = observation_date  (DATE)
--   $2 = customer_id       (UUID)
--
-- LEAKAGE GUARANTEE:
--   Every sub-query uses strict < or <= observation_date.
--   churn_labels is NEVER referenced here.
-- ============================================================

WITH base AS (
    SELECT
        c.customer_id,
        c.signup_date
    FROM curated.customers c
    WHERE c.customer_id = $2
),

tenure AS (
    SELECT
        customer_id,
        GREATEST(
            0,
            DATE_PART('year',  AGE($1::DATE, signup_date))::INTEGER * 12 +
            DATE_PART('month', AGE($1::DATE, signup_date))::INTEGER
        ) AS tenure_months
    FROM base
),

latest_sub AS (
    SELECT DISTINCT ON (customer_id)
        customer_id,
        contract_type::TEXT AS contract_type
    FROM curated.subscriptions
    WHERE customer_id = $2
      AND start_date  <= $1::DATE
      AND (end_date IS NULL OR end_date > $1::DATE)
    ORDER BY customer_id, start_date DESC
),

latest_billing AS (
    SELECT DISTINCT ON (customer_id)
        customer_id,
        monthly_charges,
        total_charges,
        payment_method::TEXT AS payment_method,
        paperless_billing,
        days_overdue        AS days_overdue_last
    FROM curated.billing
    WHERE customer_id        = $2
      AND billing_period_end < $1::DATE            -- STRICT: no future billing
    ORDER BY customer_id, billing_period_end DESC
),

billing_3m AS (
    SELECT
        customer_id,
        AVG(monthly_charges) AS avg_monthly_charges_3m,
        -- % change: (last_month_charge - earliest_in_window) / earliest_in_window
        CASE
            WHEN COUNT(*) >= 2 THEN
                (MAX(monthly_charges) - MIN(monthly_charges))::NUMERIC /
                NULLIF(MIN(monthly_charges), 0)
        END AS charge_change_pct_3m
    FROM curated.billing
    WHERE customer_id        = $2
      AND billing_period_end  < $1::DATE           -- STRICT
      AND billing_period_start >= ($1::DATE - INTERVAL '3 months')
    GROUP BY customer_id
),

latest_service AS (
    SELECT DISTINCT ON (customer_id)
        customer_id,
        phone_service,
        internet_service,
        (
            COALESCE(online_security::INT,    0) +
            COALESCE(online_backup::INT,      0) +
            COALESCE(device_protection::INT,  0) +
            COALESCE(tech_support::INT,       0) +
            COALESCE(streaming_tv::INT,       0) +
            COALESCE(streaming_movies::INT,   0)
        ) AS addon_count
    FROM curated.services
    WHERE customer_id   = $2
      AND effective_from <= $1::DATE               -- SCD-2 join
      AND (effective_to IS NULL OR effective_to > $1::DATE)
    ORDER BY customer_id, effective_from DESC
),

support_base AS (
    SELECT
        customer_id,
        ticket_category,
        opened_at,
        csat_score
    FROM curated.support_events
    WHERE customer_id = $2
      AND opened_at   < $1::TIMESTAMPTZ            -- STRICT: no future events
),

support_agg AS (
    SELECT
        customer_id,

        -- Recency counts
        COUNT(*) FILTER (
            WHERE opened_at >= ($1::TIMESTAMPTZ - INTERVAL '30 days')
        )                                                               AS tickets_30d,

        COUNT(*) FILTER (
            WHERE opened_at >= ($1::TIMESTAMPTZ - INTERVAL '90 days')
        )                                                               AS tickets_90d,

        COUNT(*)                                                        AS tickets_total,

        -- Days since last ticket (NULL if never)
        EXTRACT(DAY FROM ($1::TIMESTAMPTZ - MAX(opened_at)))::INTEGER   AS days_since_last_ticket,

        -- Category mix over last 90 days
        COUNT(*) FILTER (
            WHERE ticket_category = 'billing'
              AND opened_at >= ($1::TIMESTAMPTZ - INTERVAL '90 days')
        )::NUMERIC /
        NULLIF(
            COUNT(*) FILTER (
                WHERE opened_at >= ($1::TIMESTAMPTZ - INTERVAL '90 days')
            ), 0
        )                                                               AS pct_billing_tickets_90d,

        COUNT(*) FILTER (
            WHERE ticket_category = 'cancellation_intent'
              AND opened_at >= ($1::TIMESTAMPTZ - INTERVAL '90 days')
        )::NUMERIC /
        NULLIF(
            COUNT(*) FILTER (
                WHERE opened_at >= ($1::TIMESTAMPTZ - INTERVAL '90 days')
            ), 0
        )                                                               AS pct_cancel_tickets_90d,

        -- Average CSAT over last 90 days
        AVG(csat_score) FILTER (
            WHERE opened_at >= ($1::TIMESTAMPTZ - INTERVAL '90 days')
        )                                                               AS avg_csat_90d

    FROM support_base
    GROUP BY customer_id
)

SELECT
    -- Tenure
    t.tenure_months,

    -- Billing
    lb.monthly_charges,
    lb.total_charges,
    b3.avg_monthly_charges_3m,
    b3.charge_change_pct_3m,
    COALESCE(lb.days_overdue_last, 0)               AS days_overdue_last,

    -- Contract / Payment
    COALESCE(ls.contract_type, 'month-to-month')    AS contract_type,
    lb.payment_method,
    COALESCE(lb.paperless_billing, FALSE)            AS paperless_billing,

    -- Service flags
    COALESCE(lsv.phone_service, FALSE)              AS phone_service,
    lsv.internet_service,
    COALESCE(lsv.addon_count, 0)                    AS addon_count,

    -- Support
    COALESCE(sa.tickets_30d, 0)                     AS tickets_30d,
    COALESCE(sa.tickets_90d, 0)                     AS tickets_90d,
    COALESCE(sa.tickets_total, 0)                   AS tickets_total,
    sa.days_since_last_ticket,
    sa.pct_billing_tickets_90d,
    sa.pct_cancel_tickets_90d,
    sa.avg_csat_90d,

    -- Interaction features
    lb.monthly_charges * t.tenure_months            AS charge_x_tenure,
    COALESCE(lb.days_overdue_last, 0) *
        COALESCE(sa.pct_cancel_tickets_90d, 0)      AS overdue_x_cancel_intent

FROM tenure          t
LEFT JOIN latest_sub     ls   ON ls.customer_id  = t.customer_id
LEFT JOIN latest_billing lb   ON lb.customer_id  = t.customer_id
LEFT JOIN billing_3m     b3   ON b3.customer_id  = t.customer_id
LEFT JOIN latest_service lsv  ON lsv.customer_id = t.customer_id
LEFT JOIN support_agg    sa   ON sa.customer_id  = t.customer_id;
