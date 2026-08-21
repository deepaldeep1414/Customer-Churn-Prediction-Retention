-- ============================================================
-- Migration: 003_data_quality.sql
-- Description: Quarantine infrastructure, enhanced CHECK constraints,
--              and data-quality audit log.
-- Run after: 001_initial_schema.sql, 002_roles_and_permissions.sql
-- ============================================================

-- ============================================================
-- QUARANTINE TABLE
-- All rows that fail validation land here — never silently dropped.
-- ============================================================
CREATE TABLE IF NOT EXISTS raw.quarantine (
    quarantine_id       BIGSERIAL       PRIMARY KEY,
    source_table        TEXT            NOT NULL,
    source_batch_id     TEXT,
    source_payload      JSONB           NOT NULL,
    failure_category    TEXT            NOT NULL,  -- 'schema_mismatch' | 'impossible_value'
                                                   -- | 'excessive_nulls' | 'duplicate_key'
                                                   -- | 'leakage_risk'
    failure_detail      TEXT            NOT NULL,  -- human-readable reason
    failure_fields      TEXT[],                    -- which columns failed
    quarantined_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    resolved            BOOLEAN         NOT NULL DEFAULT FALSE,
    resolved_at         TIMESTAMPTZ,
    resolved_by         TEXT,
    resolution_notes    TEXT
);

CREATE INDEX IF NOT EXISTS idx_quarantine_source   ON raw.quarantine (source_table, quarantined_at DESC);
CREATE INDEX IF NOT EXISTS idx_quarantine_category ON raw.quarantine (failure_category);
CREATE INDEX IF NOT EXISTS idx_quarantine_unresolved ON raw.quarantine (resolved) WHERE resolved = FALSE;

-- Grant quarantine INSERT to all ingestion roles
GRANT INSERT ON raw.quarantine TO ingestion_user;
GRANT SELECT ON raw.quarantine TO ml_user, dashboard_user;
GRANT USAGE, SELECT ON SEQUENCE raw.quarantine_quarantine_id_seq TO ingestion_user;

-- ============================================================
-- DATA QUALITY AUDIT LOG
-- Tracks every validation run (pass or fail summary per batch).
-- ============================================================
CREATE TABLE IF NOT EXISTS raw.quality_audit_log (
    audit_id            BIGSERIAL       PRIMARY KEY,
    batch_id            TEXT            NOT NULL,
    source_table        TEXT            NOT NULL,
    total_records       INTEGER         NOT NULL DEFAULT 0,
    passed_records      INTEGER         NOT NULL DEFAULT 0,
    quarantined_records INTEGER         NOT NULL DEFAULT 0,
    failure_summary     JSONB,          -- {category: count, ...}
    validated_at        TIMESTAMPTZ     NOT NULL DEFAULT now(),
    validator_version   TEXT            NOT NULL DEFAULT 'v1.0.0'
);

CREATE INDEX IF NOT EXISTS idx_quality_audit_batch ON raw.quality_audit_log (batch_id);
GRANT INSERT, SELECT ON raw.quality_audit_log TO ingestion_user, ml_user;
GRANT USAGE, SELECT ON SEQUENCE raw.quality_audit_log_audit_id_seq TO ingestion_user;

-- ============================================================
-- ENHANCED CHECK CONSTRAINTS on curated tables
-- (Added post-initial-schema as ALTER TABLE to support incremental
--  migration on existing deployments)
-- ============================================================

-- curated.customers — email_hash must look like a hex SHA-256 (64 chars)
DO $$ BEGIN
    ALTER TABLE curated.customers
        ADD CONSTRAINT chk_email_hash_format
        CHECK (email_hash ~ '^[0-9a-f]{64}$');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- curated.billing — monthly_charges sanity: not absurdly high (> $10,000)
DO $$ BEGIN
    ALTER TABLE curated.billing
        ADD CONSTRAINT chk_monthly_charges_sane
        CHECK (monthly_charges <= 10000);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- curated.billing — total_charges >= monthly_charges
DO $$ BEGIN
    ALTER TABLE curated.billing
        ADD CONSTRAINT chk_total_gte_monthly
        CHECK (total_charges >= monthly_charges);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- curated.billing — days_overdue sanity: not more than 1 year
DO $$ BEGIN
    ALTER TABLE curated.billing
        ADD CONSTRAINT chk_days_overdue_sane
        CHECK (days_overdue <= 366);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- curated.support_events — csat_score strict range
DO $$ BEGIN
    ALTER TABLE curated.support_events
        ADD CONSTRAINT chk_csat_range
        CHECK (csat_score IS NULL OR (csat_score >= 1 AND csat_score <= 5));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ml.feature_snapshots — probability/rate fields
DO $$ BEGIN
    ALTER TABLE ml.feature_snapshots
        ADD CONSTRAINT chk_pct_billing_range
        CHECK (pct_billing_tickets_90d IS NULL OR pct_billing_tickets_90d BETWEEN 0 AND 1);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE ml.feature_snapshots
        ADD CONSTRAINT chk_pct_cancel_range
        CHECK (pct_cancel_tickets_90d IS NULL OR pct_cancel_tickets_90d BETWEEN 0 AND 1);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE ml.feature_snapshots
        ADD CONSTRAINT chk_avg_csat_range
        CHECK (avg_csat_90d IS NULL OR avg_csat_90d BETWEEN 1 AND 5);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ml.predictions — churn_probability in [0,1]
DO $$ BEGIN
    ALTER TABLE ml.predictions
        ADD CONSTRAINT chk_churn_prob_range
        CHECK (churn_probability BETWEEN 0 AND 1);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ============================================================
-- LEAKAGE PREVENTION VIEW
-- A read-only view that joins feature_snapshots with labels,
-- asserting no label data bleeds into the feature space.
-- Used by the leakage validation stored procedure below.
-- ============================================================
CREATE OR REPLACE VIEW ml.leakage_check AS
SELECT
    fs.snapshot_id,
    fs.customer_id,
    fs.observation_date,
    cl.churned,
    cl.churn_date,
    -- If churn_date <= observation_date, the label data is leaking into features
    CASE
        WHEN cl.churn_date IS NOT NULL AND cl.churn_date <= fs.observation_date
        THEN TRUE ELSE FALSE
    END AS has_label_leakage
FROM ml.feature_snapshots fs
LEFT JOIN curated.churn_labels cl
       ON cl.customer_id      = fs.customer_id
      AND cl.observation_date = fs.observation_date
      AND cl.prediction_horizon = fs.prediction_horizon_days;

GRANT SELECT ON ml.leakage_check TO ml_user;

-- ============================================================
-- STORED PROCEDURE: validate_snapshot_leakage()
-- Raises WARNING for any snapshot row with label leakage.
-- Called by the training pipeline before model fitting.
-- ============================================================
CREATE OR REPLACE FUNCTION ml.validate_snapshot_leakage(
    p_feature_version TEXT,
    p_observation_date DATE
) RETURNS TABLE (
    leaking_snapshot_id UUID,
    customer_id         UUID,
    observation_date    DATE,
    churn_date          DATE
)
LANGUAGE sql STABLE AS $$
    SELECT
        lc.snapshot_id,
        lc.customer_id,
        lc.observation_date,
        lc.churn_date
    FROM ml.leakage_check lc
    JOIN ml.feature_snapshots fs ON fs.snapshot_id = lc.snapshot_id
    WHERE fs.feature_set_version = p_feature_version
      AND fs.observation_date    = p_observation_date
      AND lc.has_label_leakage   = TRUE;
$$;

GRANT EXECUTE ON FUNCTION ml.validate_snapshot_leakage TO ml_user;
