-- ============================================================
-- Migration: 001_initial_schema.sql
-- Description: Bootstrap raw, curated, and ml schemas with all
--              core tables, types, indexes, and constraints.
-- ============================================================

-- ============================================================
-- SCHEMAS
-- ============================================================
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS curated;
CREATE SCHEMA IF NOT EXISTS ml;

-- ============================================================
-- ENUM TYPES
-- ============================================================
DO $$ BEGIN
    CREATE TYPE curated.contract_type   AS ENUM ('month-to-month', 'one_year', 'two_year');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE curated.payment_method  AS ENUM ('electronic_check', 'mailed_check', 'bank_transfer', 'credit_card');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE curated.risk_band       AS ENUM ('low', 'medium', 'high', 'critical');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE curated.ticket_category AS ENUM ('billing', 'technical', 'general', 'cancellation_intent');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE curated.ticket_status   AS ENUM ('open', 'in_progress', 'resolved', 'closed');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE ml.model_stage          AS ENUM ('staging', 'champion', 'archived');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ============================================================
-- 1. CURATED.CUSTOMERS
-- ============================================================
CREATE TABLE IF NOT EXISTS curated.customers (
    customer_id         UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id         TEXT            NOT NULL UNIQUE,
    first_name          TEXT,
    last_name           TEXT,
    email_hash          TEXT            NOT NULL,         -- SHA-256(lower(email))
    phone_hash          TEXT,
    gender              TEXT            CHECK (gender IN ('male', 'female', 'other', 'unknown')),
    senior_citizen      BOOLEAN         NOT NULL DEFAULT FALSE,
    partner             BOOLEAN         NOT NULL DEFAULT FALSE,
    dependents          BOOLEAN         NOT NULL DEFAULT FALSE,
    signup_date         DATE            NOT NULL,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_customers_signup_date ON curated.customers (signup_date);
CREATE INDEX IF NOT EXISTS idx_customers_external_id ON curated.customers (external_id);

-- ============================================================
-- 2. CURATED.SUBSCRIPTIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS curated.subscriptions (
    subscription_id     UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id         UUID            NOT NULL REFERENCES curated.customers (customer_id) ON DELETE CASCADE,
    contract_type       curated.contract_type NOT NULL,
    start_date          DATE            NOT NULL,
    end_date            DATE,
    is_active           BOOLEAN         NOT NULL GENERATED ALWAYS AS (end_date IS NULL) STORED,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    CONSTRAINT chk_sub_dates CHECK (end_date IS NULL OR end_date >= start_date)
);

CREATE INDEX IF NOT EXISTS idx_subscriptions_customer_id ON curated.subscriptions (customer_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_active      ON curated.subscriptions (customer_id) WHERE is_active = TRUE;

-- ============================================================
-- 3. CURATED.BILLING
-- ============================================================
CREATE TABLE IF NOT EXISTS curated.billing (
    billing_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id             UUID            NOT NULL REFERENCES curated.customers (customer_id) ON DELETE CASCADE,
    billing_period_start    DATE            NOT NULL,
    billing_period_end      DATE            NOT NULL,
    monthly_charges         NUMERIC(10, 2)  NOT NULL CHECK (monthly_charges >= 0),
    total_charges           NUMERIC(12, 2)  NOT NULL CHECK (total_charges >= 0),
    payment_method          curated.payment_method NOT NULL,
    paperless_billing       BOOLEAN         NOT NULL DEFAULT FALSE,
    payment_date            DATE,
    days_overdue            INTEGER         NOT NULL DEFAULT 0 CHECK (days_overdue >= 0),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    CONSTRAINT chk_billing_period CHECK (billing_period_end > billing_period_start)
);

CREATE INDEX IF NOT EXISTS idx_billing_customer_id ON curated.billing (customer_id);
CREATE INDEX IF NOT EXISTS idx_billing_period      ON curated.billing (customer_id, billing_period_start DESC);

-- ============================================================
-- 4. CURATED.SERVICES
-- ============================================================
CREATE TABLE IF NOT EXISTS curated.services (
    service_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id             UUID            NOT NULL REFERENCES curated.customers (customer_id) ON DELETE CASCADE,
    phone_service           BOOLEAN         NOT NULL DEFAULT FALSE,
    multiple_lines          BOOLEAN,
    internet_service        TEXT            CHECK (internet_service IN ('dsl', 'fiber_optic', 'none')),
    online_security         BOOLEAN,
    online_backup           BOOLEAN,
    device_protection       BOOLEAN,
    tech_support            BOOLEAN,
    streaming_tv            BOOLEAN,
    streaming_movies        BOOLEAN,
    effective_from          DATE            NOT NULL,
    effective_to            DATE,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    CONSTRAINT chk_service_dates CHECK (effective_to IS NULL OR effective_to > effective_from)
);

CREATE INDEX IF NOT EXISTS idx_services_customer_id ON curated.services (customer_id);
CREATE INDEX IF NOT EXISTS idx_services_active      ON curated.services (customer_id) WHERE effective_to IS NULL;

-- ============================================================
-- 5. CURATED.SUPPORT_EVENTS
-- ============================================================
CREATE TABLE IF NOT EXISTS curated.support_events (
    event_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id             UUID            NOT NULL REFERENCES curated.customers (customer_id) ON DELETE CASCADE,
    ticket_category         curated.ticket_category NOT NULL,
    ticket_status           curated.ticket_status   NOT NULL DEFAULT 'open',
    opened_at               TIMESTAMPTZ     NOT NULL,
    resolved_at             TIMESTAMPTZ,
    resolution_days         INTEGER         GENERATED ALWAYS AS (
                                CASE WHEN resolved_at IS NOT NULL
                                     THEN EXTRACT(DAY FROM resolved_at - opened_at)::INTEGER
                                END
                            ) STORED,
    csat_score              SMALLINT        CHECK (csat_score BETWEEN 1 AND 5),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    CONSTRAINT chk_resolved_after_opened CHECK (resolved_at IS NULL OR resolved_at >= opened_at)
);

CREATE INDEX IF NOT EXISTS idx_support_customer_id ON curated.support_events (customer_id);
CREATE INDEX IF NOT EXISTS idx_support_opened_at   ON curated.support_events (customer_id, opened_at DESC);

-- ============================================================
-- 6. CURATED.CHURN_LABELS
-- ============================================================
CREATE TABLE IF NOT EXISTS curated.churn_labels (
    label_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id             UUID            NOT NULL REFERENCES curated.customers (customer_id) ON DELETE CASCADE,
    observation_date        DATE            NOT NULL,
    prediction_horizon      INTEGER         NOT NULL DEFAULT 30 CHECK (prediction_horizon > 0),
    churned                 BOOLEAN         NOT NULL,
    churn_date              DATE,
    labeling_method         TEXT            NOT NULL DEFAULT 'subscription_end',
    label_version           INTEGER         NOT NULL DEFAULT 1,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    UNIQUE (customer_id, observation_date, prediction_horizon, label_version),
    CONSTRAINT chk_churn_date CHECK (
        churned = FALSE OR (churn_date IS NOT NULL AND churn_date > observation_date)
    )
);

CREATE INDEX IF NOT EXISTS idx_labels_observation ON curated.churn_labels (observation_date);
CREATE INDEX IF NOT EXISTS idx_labels_customer    ON curated.churn_labels (customer_id);

-- ============================================================
-- 7. ML.FEATURE_SNAPSHOTS
-- ============================================================
CREATE TABLE IF NOT EXISTS ml.feature_snapshots (
    snapshot_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id             UUID            NOT NULL REFERENCES curated.customers (customer_id) ON DELETE CASCADE,
    observation_date        DATE            NOT NULL,
    prediction_horizon_days INTEGER         NOT NULL DEFAULT 30,
    feature_set_version     TEXT            NOT NULL,
    -- Tenure
    tenure_months           INTEGER         NOT NULL CHECK (tenure_months >= 0),
    -- Billing
    monthly_charges         NUMERIC(10, 2)  NOT NULL,
    total_charges           NUMERIC(12, 2)  NOT NULL,
    avg_monthly_charges_3m  NUMERIC(10, 2),
    charge_change_pct_3m    NUMERIC(8, 4),
    days_overdue_last       INTEGER         NOT NULL DEFAULT 0,
    -- Contract / Payment
    contract_type           TEXT            NOT NULL,
    payment_method          TEXT            NOT NULL,
    paperless_billing       BOOLEAN         NOT NULL,
    -- Service flags
    phone_service           BOOLEAN         NOT NULL,
    internet_service        TEXT,
    addon_count             INTEGER         NOT NULL DEFAULT 0,
    -- Support
    tickets_30d             INTEGER         NOT NULL DEFAULT 0,
    tickets_90d             INTEGER         NOT NULL DEFAULT 0,
    tickets_total           INTEGER         NOT NULL DEFAULT 0,
    days_since_last_ticket  INTEGER,
    pct_billing_tickets_90d NUMERIC(5, 4),
    pct_cancel_tickets_90d  NUMERIC(5, 4),
    avg_csat_90d            NUMERIC(4, 2),
    -- Interaction features
    charge_x_tenure         NUMERIC(14, 4),
    overdue_x_cancel_intent NUMERIC(8, 4),
    -- Metadata
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    UNIQUE (customer_id, observation_date, prediction_horizon_days, feature_set_version)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_obs_date ON ml.feature_snapshots (observation_date);
CREATE INDEX IF NOT EXISTS idx_snapshots_customer  ON ml.feature_snapshots (customer_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_version   ON ml.feature_snapshots (feature_set_version);

-- ============================================================
-- 8. ML.PREDICTIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS ml.predictions (
    prediction_id       UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id         UUID            NOT NULL REFERENCES curated.customers (customer_id) ON DELETE CASCADE,
    model_version       TEXT            NOT NULL,
    snapshot_id         UUID            REFERENCES ml.feature_snapshots (snapshot_id),
    observation_date    DATE            NOT NULL,
    churn_probability   NUMERIC(6, 5)   NOT NULL CHECK (churn_probability BETWEEN 0 AND 1),
    risk_band           curated.risk_band NOT NULL,
    threshold_used      NUMERIC(6, 5)   NOT NULL,
    scored_at           TIMESTAMPTZ     NOT NULL DEFAULT now(),
    top_drivers         JSONB,
    request_source      TEXT            NOT NULL DEFAULT 'batch',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_predictions_customer  ON ml.predictions (customer_id, scored_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model     ON ml.predictions (model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_risk_band ON ml.predictions (risk_band, scored_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_obs_date  ON ml.predictions (observation_date);

-- GIN index for JSONB top_drivers querying
CREATE INDEX IF NOT EXISTS idx_predictions_drivers ON ml.predictions USING GIN (top_drivers);

-- ============================================================
-- 9. ML.MODEL_REGISTRY
-- ============================================================
CREATE TABLE IF NOT EXISTS ml.model_registry (
    registry_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name                  TEXT            NOT NULL,
    model_version               TEXT            NOT NULL UNIQUE,
    algorithm                   TEXT            NOT NULL,
    stage                       ml.model_stage  NOT NULL DEFAULT 'staging',
    mlflow_run_id               TEXT,
    mlflow_model_uri            TEXT,
    feature_set_version         TEXT            NOT NULL,
    train_dataset_hash          TEXT            NOT NULL,
    val_dataset_hash            TEXT            NOT NULL,
    test_dataset_hash           TEXT,
    -- Frozen evaluation metrics (test set)
    pr_auc                      NUMERIC(6, 5),
    roc_auc                     NUMERIC(6, 5),
    brier_score                 NUMERIC(6, 5),
    recall_at_threshold         NUMERIC(6, 5),
    precision_at_threshold      NUMERIC(6, 5),
    f1_at_threshold             NUMERIC(6, 5),
    threshold                   NUMERIC(6, 5),
    -- Promotion audit
    promoted_by                 TEXT,
    promoted_at                 TIMESTAMPTZ,
    promotion_notes             TEXT,
    archived_at                 TIMESTAMPTZ,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT now(),
    -- Enforce single champion at a time
    CONSTRAINT single_champion EXCLUDE USING btree (stage WITH =)
        WHERE (stage = 'champion')
        DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX IF NOT EXISTS idx_registry_stage   ON ml.model_registry (stage);
CREATE INDEX IF NOT EXISTS idx_registry_version ON ml.model_registry (model_version);

-- ============================================================
-- RAW LAYER (Ingestion Staging)
-- ============================================================
CREATE TABLE IF NOT EXISTS raw.customers_raw (
    ingest_id       BIGSERIAL   PRIMARY KEY,
    source_system   TEXT        NOT NULL,
    payload         JSONB       NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    batch_id        TEXT        NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_raw_customers_batch ON raw.customers_raw (batch_id);

CREATE TABLE IF NOT EXISTS raw.billing_raw (
    ingest_id       BIGSERIAL   PRIMARY KEY,
    source_system   TEXT        NOT NULL,
    payload         JSONB       NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    batch_id        TEXT        NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_raw_billing_batch ON raw.billing_raw (batch_id);

CREATE TABLE IF NOT EXISTS raw.support_raw (
    ingest_id       BIGSERIAL   PRIMARY KEY,
    source_system   TEXT        NOT NULL,
    payload         JSONB       NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    batch_id        TEXT        NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_raw_support_batch ON raw.support_raw (batch_id);
