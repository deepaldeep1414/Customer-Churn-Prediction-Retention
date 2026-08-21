-- ============================================================
-- Migration: 002_roles_and_permissions.sql
-- Description: Least-privilege DB roles for each service.
-- Run as superuser (churn_admin) after 001_initial_schema.sql
-- ============================================================

-- NOTE: Passwords are injected at runtime via psql \set or
--       passed as environment variables from Secrets Manager.
--       Never commit plaintext passwords.

-- ============================================================
-- API SERVICE ROLE
-- Read curated tables; read+write ml.predictions
-- ============================================================
DO $$ BEGIN
    CREATE ROLE api_user LOGIN;
EXCEPTION WHEN duplicate_object THEN
    RAISE NOTICE 'Role api_user already exists, skipping.';
END $$;

GRANT CONNECT ON DATABASE churn_db TO api_user;
GRANT USAGE   ON SCHEMA curated, ml TO api_user;

GRANT SELECT ON
    curated.customers,
    curated.subscriptions,
    curated.billing,
    curated.services,
    curated.support_events
TO api_user;

GRANT SELECT, INSERT ON ml.predictions        TO api_user;
GRANT SELECT         ON ml.feature_snapshots  TO api_user;
GRANT SELECT         ON ml.model_registry     TO api_user;

-- Sequence access for predictions (if BIGSERIAL ever added)
ALTER DEFAULT PRIVILEGES IN SCHEMA ml
    GRANT SELECT, USAGE ON SEQUENCES TO api_user;

-- ============================================================
-- ML TRAINING ROLE
-- Full read on curated; read+write on ml schema
-- ============================================================
DO $$ BEGIN
    CREATE ROLE ml_user LOGIN;
EXCEPTION WHEN duplicate_object THEN
    RAISE NOTICE 'Role ml_user already exists, skipping.';
END $$;

GRANT CONNECT ON DATABASE churn_db TO ml_user;
GRANT USAGE   ON SCHEMA curated, ml TO ml_user;

GRANT SELECT ON ALL TABLES IN SCHEMA curated TO ml_user;

GRANT SELECT, INSERT, UPDATE ON
    ml.feature_snapshots,
    ml.model_registry
TO ml_user;

GRANT SELECT, INSERT ON curated.churn_labels TO ml_user;
GRANT SELECT         ON ml.predictions        TO ml_user;

ALTER DEFAULT PRIVILEGES IN SCHEMA curated
    GRANT SELECT ON TABLES TO ml_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA ml
    GRANT SELECT, INSERT, UPDATE ON TABLES TO ml_user;

-- ============================================================
-- INGESTION ROLE
-- Write-only to raw schema; no curated/ml access
-- ============================================================
DO $$ BEGIN
    CREATE ROLE ingestion_user LOGIN;
EXCEPTION WHEN duplicate_object THEN
    RAISE NOTICE 'Role ingestion_user already exists, skipping.';
END $$;

GRANT CONNECT ON DATABASE churn_db     TO ingestion_user;
GRANT USAGE   ON SCHEMA raw            TO ingestion_user;
GRANT INSERT  ON ALL TABLES IN SCHEMA raw TO ingestion_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA raw TO ingestion_user;

-- Explicitly revoke any curated/ml access
REVOKE ALL ON SCHEMA curated FROM ingestion_user;
REVOKE ALL ON SCHEMA ml      FROM ingestion_user;

ALTER DEFAULT PRIVILEGES IN SCHEMA raw
    GRANT INSERT ON TABLES TO ingestion_user;

-- ============================================================
-- DASHBOARD ROLE (Superset / BI read-only)
-- Read curated + ml.predictions + ml.model_registry
-- NO access to raw schema; NO PII fields via view layer
-- ============================================================
DO $$ BEGIN
    CREATE ROLE dashboard_user LOGIN;
EXCEPTION WHEN duplicate_object THEN
    RAISE NOTICE 'Role dashboard_user already exists, skipping.';
END $$;

GRANT CONNECT ON DATABASE churn_db TO dashboard_user;
GRANT USAGE   ON SCHEMA curated, ml TO dashboard_user;

-- Grant view-level access (not raw PII columns — those are restricted via views below)
GRANT SELECT ON
    curated.subscriptions,
    curated.billing,
    curated.services,
    curated.support_events,
    curated.churn_labels
TO dashboard_user;

GRANT SELECT ON
    ml.predictions,
    ml.model_registry,
    ml.feature_snapshots
TO dashboard_user;

-- Explicitly revoke raw schema
REVOKE ALL ON SCHEMA raw FROM dashboard_user;

-- PII-safe customer view (no name, no hash, just pseudonymous ID)
CREATE OR REPLACE VIEW curated.customers_safe AS
    SELECT
        customer_id,
        gender,
        senior_citizen,
        partner,
        dependents,
        signup_date,
        created_at
    FROM curated.customers;

GRANT SELECT ON curated.customers_safe TO dashboard_user;

-- ============================================================
-- MLFLOW BACKEND ROLE
-- Needs full access to mlflow tracking tables (own schema)
-- ============================================================
DO $$ BEGIN
    CREATE ROLE mlflow_user LOGIN;
EXCEPTION WHEN duplicate_object THEN
    RAISE NOTICE 'Role mlflow_user already exists, skipping.';
END $$;

GRANT CONNECT  ON DATABASE churn_db TO mlflow_user;
-- MLflow creates its own tables; grant schema-level creation
GRANT CREATE, USAGE ON SCHEMA public TO mlflow_user;
