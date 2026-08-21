-- ==============================================================
-- apps/dashboard/dashboards/churn_queries.sql
-- Superset chart SQL queries — one section per chart.
-- Import via: Superset → SQL Lab → Save As Chart → Add to Dashboard
--
-- Prerequisites:
--   - Database connection named "churn_db" pointing to PostgreSQL
--   - dashboard_user role has been granted (see 002_roles_and_permissions.sql)
-- ==============================================================


-- ============================================================
-- CHART 1 — Executive Overview: Current Churn Risk Summary
-- Dashboard: Executive Overview
-- Chart type: Big Number with Trendline (one per metric)
-- ============================================================
/* Title: Total Customers Scored Today */
SELECT
    COUNT(DISTINCT customer_id)                                   AS total_customers_scored,
    COUNT(*) FILTER (WHERE risk_band = 'critical')                AS critical_count,
    COUNT(*) FILTER (WHERE risk_band = 'high')                    AS high_count,
    COUNT(*) FILTER (WHERE risk_band = 'medium')                  AS medium_count,
    COUNT(*) FILTER (WHERE risk_band = 'low')                     AS low_count,
    ROUND(AVG(churn_probability)::NUMERIC, 3)                     AS avg_churn_prob,
    ROUND(
        COUNT(*) FILTER (WHERE churn_probability >= threshold_used)::NUMERIC /
        NULLIF(COUNT(*), 0) * 100, 1
    )                                                             AS predicted_churn_rate_pct
FROM ml.predictions
WHERE observation_date = CURRENT_DATE;


-- ============================================================
-- CHART 2 — Risk Distribution (Bar Chart / Pie Chart)
-- Dashboard: Executive Overview, Risk Distribution
-- ============================================================
/* Title: Risk Band Distribution by Day */
SELECT
    observation_date,
    risk_band,
    COUNT(*)                                              AS customer_count,
    ROUND(AVG(churn_probability)::NUMERIC, 3)             AS avg_prob
FROM ml.predictions
WHERE observation_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY observation_date, risk_band
ORDER BY observation_date, risk_band;


-- ============================================================
-- CHART 3 — Score Distribution (Histogram)
-- Dashboard: Risk Distribution
-- ============================================================
/* Title: Churn Probability Distribution */
SELECT
    ROUND(churn_probability::NUMERIC, 1) AS prob_bucket,
    COUNT(*)                             AS count
FROM ml.predictions
WHERE observation_date = CURRENT_DATE
GROUP BY ROUND(churn_probability::NUMERIC, 1)
ORDER BY prob_bucket;


-- ============================================================
-- CHART 4 — Segment Analysis: Risk by Contract Type
-- Dashboard: Segment Analysis
-- Chart type: Grouped Bar Chart
-- ============================================================
/* Title: Churn Risk by Contract Type */
SELECT
    ls.contract_type,
    p.risk_band,
    COUNT(*)                                              AS customer_count,
    ROUND(AVG(p.churn_probability)::NUMERIC, 3)           AS avg_churn_prob
FROM ml.predictions p
JOIN ml.feature_snapshots fs
  ON fs.snapshot_id    = p.snapshot_id
JOIN LATERAL (
    SELECT contract_type
    FROM curated.subscriptions
    WHERE customer_id = p.customer_id
      AND start_date  <= p.observation_date
      AND (end_date IS NULL OR end_date > p.observation_date)
    ORDER BY start_date DESC LIMIT 1
) ls ON TRUE
WHERE p.observation_date = CURRENT_DATE
GROUP BY ls.contract_type, p.risk_band
ORDER BY ls.contract_type, p.risk_band;


-- ============================================================
-- CHART 5 — Segment Analysis: Risk by Tenure Band
-- Dashboard: Segment Analysis
-- ============================================================
/* Title: Churn Risk by Tenure Band */
SELECT
    CASE
        WHEN fs.tenure_months <= 6   THEN '0–6 months'
        WHEN fs.tenure_months <= 12  THEN '7–12 months'
        WHEN fs.tenure_months <= 24  THEN '13–24 months'
        WHEN fs.tenure_months <= 48  THEN '25–48 months'
        ELSE '> 48 months'
    END                                                   AS tenure_band,
    COUNT(*)                                              AS customer_count,
    ROUND(AVG(p.churn_probability)::NUMERIC, 3)           AS avg_churn_prob,
    COUNT(*) FILTER (WHERE p.risk_band IN ('high','critical')) AS high_risk_count
FROM ml.predictions p
JOIN ml.feature_snapshots fs
  ON fs.snapshot_id    = p.snapshot_id
WHERE p.observation_date = CURRENT_DATE
GROUP BY 1
ORDER BY MIN(fs.tenure_months);


-- ============================================================
-- CHART 6 — Customer Explorer (Table / Drill-Down)
-- Dashboard: Customer Explorer
-- Chart type: Table with search + sort
-- Add Superset filter on risk_band, observation_date, contract_type
-- ============================================================
/* Title: Customer Risk Explorer */
SELECT
    p.customer_id,
    p.churn_probability,
    p.risk_band,
    p.threshold_used,
    p.observation_date,
    p.model_version,
    p.top_drivers,                                        -- JSONB: inspect in Superset
    fs.tenure_months,
    fs.monthly_charges,
    fs.contract_type,
    fs.tickets_90d,
    fs.pct_cancel_tickets_90d,
    fs.avg_csat_90d
FROM ml.predictions p
JOIN ml.feature_snapshots fs ON fs.snapshot_id = p.snapshot_id
WHERE p.observation_date = '{{ filter_values("observation_date")[0] | default(current_date) }}'
  AND ('{{ filter_values("risk_band")[0] | default("all") }}' = 'all'
       OR p.risk_band = '{{ filter_values("risk_band")[0] }}')
ORDER BY p.churn_probability DESC
LIMIT 500;


-- ============================================================
-- CHART 7 — Top SHAP Drivers (Aggregated)
-- Dashboard: Customer Explorer
-- Chart type: Horizontal Bar
-- ============================================================
/* Title: Most Impactful Risk Drivers (Aggregated SHAP) */
WITH driver_rows AS (
    SELECT
        p.observation_date,
        jsonb_array_elements(p.top_drivers) AS driver
    FROM ml.predictions p
    WHERE p.observation_date = CURRENT_DATE
      AND p.top_drivers IS NOT NULL
),
driver_extracted AS (
    SELECT
        observation_date,
        driver->>'label'         AS driver_label,
        (driver->>'shap_value')::FLOAT AS shap_value,
        driver->>'direction'     AS direction
    FROM driver_rows
)
SELECT
    driver_label,
    ROUND(AVG(ABS(shap_value))::NUMERIC, 4)   AS mean_abs_shap,
    COUNT(*) FILTER (WHERE direction = 'increases_risk')  AS increases_risk_count,
    COUNT(*) FILTER (WHERE direction = 'decreases_risk')  AS decreases_risk_count,
    COUNT(*)                                               AS total_appearances
FROM driver_extracted
GROUP BY driver_label
ORDER BY mean_abs_shap DESC
LIMIT 15;


-- ============================================================
-- CHART 8 — Model Monitoring: PR-AUC over time
-- Dashboard: Model Monitoring
-- Chart type: Line Chart (one line per model_version)
-- ============================================================
/* Title: Model PR-AUC Trend */
SELECT
    mr.model_version,
    mr.algorithm,
    mr.pr_auc,
    mr.promoted_at::DATE AS promoted_date
FROM ml.model_registry mr
WHERE mr.stage IN ('champion', 'archived')
ORDER BY mr.promoted_at;


-- ============================================================
-- CHART 9 — Model Monitoring: Prediction Volume
-- Dashboard: Model Monitoring
-- ============================================================
/* Title: Daily Prediction Volume */
SELECT
    observation_date,
    model_version,
    COUNT(*)                                                   AS predictions_count,
    COUNT(*) FILTER (WHERE risk_band = 'critical')             AS critical_count,
    COUNT(*) FILTER (WHERE risk_band = 'high')                 AS high_count,
    ROUND(AVG(churn_probability)::NUMERIC, 3)                  AS avg_prob,
    ROUND(STDDEV(churn_probability)::NUMERIC, 3)               AS std_prob
FROM ml.predictions
WHERE observation_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY observation_date, model_version
ORDER BY observation_date;


-- ============================================================
-- CHART 10 — Model Monitoring: Score Distribution Drift
-- Dashboard: Drift Monitoring
-- Chart type: Box Plot (by week)
-- ============================================================
/* Title: Churn Score Distribution Over Time */
SELECT
    DATE_TRUNC('week', observation_date::TIMESTAMPTZ)::DATE  AS week_start,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY churn_probability)::NUMERIC, 3) AS p25,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY churn_probability)::NUMERIC, 3) AS median,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY churn_probability)::NUMERIC, 3) AS p75,
    ROUND(AVG(churn_probability)::NUMERIC, 3)                AS mean,
    COUNT(*)                                                  AS n
FROM ml.predictions
WHERE observation_date >= CURRENT_DATE - INTERVAL '180 days'
GROUP BY 1
ORDER BY 1;


-- ============================================================
-- CHART 11 — Drift Monitoring: Feature PSI Table
-- Dashboard: Drift Monitoring
-- Chart type: Table with conditional formatting
-- ============================================================
/* Title: PSI by Feature (Reference vs Current Month) */
WITH reference AS (
    SELECT
        tenure_months, monthly_charges, total_charges,
        tickets_90d, pct_cancel_tickets_90d, days_overdue_last
    FROM ml.feature_snapshots
    WHERE observation_date = (CURRENT_DATE - INTERVAL '30 days')::DATE
      AND feature_set_version = 'v1.0.0'
),
current AS (
    SELECT
        tenure_months, monthly_charges, total_charges,
        tickets_90d, pct_cancel_tickets_90d, days_overdue_last
    FROM ml.feature_snapshots
    WHERE observation_date = CURRENT_DATE
      AND feature_set_version = 'v1.0.0'
)
-- PSI computation at the SQL level uses mean comparison as a proxy
-- Full PSI requires Python (monitoring_pipeline.py). This view shows
-- statistical distance as a monitoring signal.
SELECT
    'tenure_months'         AS feature,
    ROUND(ABS(AVG(r.tenure_months)   - AVG(c.tenure_months))   / NULLIF(STDDEV(r.tenure_months), 0.001), 3)   AS standardised_diff
FROM reference r, current c
UNION ALL
SELECT
    'monthly_charges',
    ROUND(ABS(AVG(r.monthly_charges) - AVG(c.monthly_charges)) / NULLIF(STDDEV(r.monthly_charges), 0.001), 3)
FROM reference r, current c
UNION ALL
SELECT
    'tickets_90d',
    ROUND(ABS(AVG(r.tickets_90d)     - AVG(c.tickets_90d))     / NULLIF(STDDEV(r.tickets_90d), 0.001), 3)
FROM reference r, current c
UNION ALL
SELECT
    'pct_cancel_tickets_90d',
    ROUND(ABS(AVG(r.pct_cancel_tickets_90d) - AVG(c.pct_cancel_tickets_90d)) / NULLIF(STDDEV(r.pct_cancel_tickets_90d), 0.001), 3)
FROM reference r, current c
ORDER BY standardised_diff DESC;


-- ============================================================
-- CHART 12 — Operations Status: Data Quality
-- Dashboard: Operations Status
-- Chart type: Big Number + Table
-- ============================================================
/* Title: Data Quality — Quarantine Rate (Last 7 days) */
SELECT
    source_table,
    SUM(total_records)          AS total_records,
    SUM(quarantined_records)    AS quarantined,
    SUM(passed_records)         AS passed,
    ROUND(
        SUM(quarantined_records)::NUMERIC /
        NULLIF(SUM(total_records), 0) * 100, 2
    )                           AS quarantine_rate_pct,
    MAX(validated_at)           AS last_batch_at
FROM raw.quality_audit_log
WHERE validated_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
GROUP BY source_table
ORDER BY quarantine_rate_pct DESC;


-- ============================================================
-- CHART 13 — Calibration: Reliability Diagram Data
-- Dashboard: Model Monitoring
-- Chart type: Line Chart (actual vs predicted)
-- ============================================================
/* Title: Model Calibration — Reliability Diagram */
WITH binned AS (
    SELECT
        p.churn_probability,
        COALESCE(cl.churned::INT, 0)                            AS actual_churned,
        ROUND(p.churn_probability::NUMERIC, 1)                  AS prob_bin
    FROM ml.predictions p
    LEFT JOIN curated.churn_labels cl
           ON cl.customer_id      = p.customer_id
          AND cl.observation_date = p.observation_date
          AND cl.prediction_horizon = 30
    WHERE p.observation_date >= CURRENT_DATE - INTERVAL '90 days'
)
SELECT
    prob_bin                              AS predicted_probability,
    ROUND(AVG(actual_churned)::NUMERIC, 3) AS actual_churn_rate,
    COUNT(*)                              AS sample_count
FROM binned
WHERE actual_churned IS NOT NULL
GROUP BY prob_bin
HAVING COUNT(*) >= 20
ORDER BY prob_bin;
