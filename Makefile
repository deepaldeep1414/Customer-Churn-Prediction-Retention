# ==============================================================
# Customer Churn Platform — Makefile
# ==============================================================

.PHONY: help up down up-training up-prod logs shell-api shell-db \
        migrate migrate-all seed validate-schema \
        test test-unit test-leakage test-smoke test-integration \
        lint format type-check \
        ingest-customers ingest-billing ingest-support \
        feature-snapshot label-data score-batch monitor \
        train train-full train-tuned \
        check-triggers promote rollback \
        metrics build push clean

# ── Docker Compose shorthand ────────────────────────────────
COMPOSE      = docker compose -f infra/docker-compose.yml
ENV_FILE     = .env

help:
	@echo ""
	@echo "  Customer Churn Platform — Available commands"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make up                  Start core services (Postgres, MLflow, API, Nginx, Superset)"
	@echo "  make up-training         Start all services including training worker"
	@echo "  make up-prod             Start production compose stack"
	@echo "  make down                Stop all services"
	@echo "  make logs                Tail logs for all services"
	@echo "  make shell-api           Open shell in API container"
	@echo "  make shell-db            Open psql in Postgres container"
	@echo "  make migrate             Run all migrations (001, 002, 003)"
	@echo "  make seed                Load sample data"
	@echo "  make test                Run all tests (unit + integration)"
	@echo "  make test-unit           Run unit tests only"
	@echo "  make test-leakage        Leakage gate (must pass 100%)"
	@echo "  make test-integration    Run integration tests"
	@echo "  make test-smoke          Run smoke tests against live API"
	@echo "  make lint                Run Ruff linter"
	@echo "  make format              Run Black + isort"
	@echo "  make type-check          Run mypy"
	@echo "  make ingest-customers    Ingest customers CSV"
	@echo "  make ingest-billing      Ingest billing CSV"
	@echo "  make ingest-support      Ingest support events CSV"
	@echo "  make feature-snapshot    Build feature snapshots for today"
	@echo "  make label-data          Generate churn labels for today"
	@echo "  make score-batch         Run daily batch scoring"
	@echo "  make monitor             Run drift monitoring vs yesterday"
	@echo "  make train               Run full training pipeline (no tuning)"
	@echo "  make train-tuned         Run training with Optuna tuning"
	@echo "  make check-triggers      Evaluate retraining trigger conditions"
	@echo "  make promote             Run promotion gates for latest staged model"
	@echo "  make rollback            Rollback to previous champion (dry-run by default)"
	@echo "  make metrics             Show API latency metrics (JSON)"
	@echo ""

# ── Core services ───────────────────────────────────────────
up:
	$(COMPOSE) --env-file $(ENV_FILE) up -d postgres mlflow api nginx superset
	@echo "Services up. API → http://localhost:8000 | MLflow → http://localhost:5000 | Superset → http://localhost:8088"

up-training:
	$(COMPOSE) --env-file $(ENV_FILE) --profile training up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

shell-api:
	$(COMPOSE) exec api /bin/sh

shell-db:
	$(COMPOSE) exec postgres psql -U churn_admin -d churn_db

# ── Production compose ──────────────────────────────────────
up-prod:
	docker compose -f infra/docker-compose.prod.yml --env-file $(ENV_FILE) up -d

# ── Database ────────────────────────────────────────────────
migrate:
	$(COMPOSE) exec postgres psql -U churn_admin -d churn_db \
		-f /docker-entrypoint-initdb.d/001_initial_schema.sql

# Run all migrations in sequence (idempotent)
migrate-all:
	$(COMPOSE) exec postgres psql -U churn_admin -d churn_db \
		-f /docker-entrypoint-initdb.d/001_initial_schema.sql
	$(COMPOSE) exec postgres psql -U churn_admin -d churn_db \
		-f /docker-entrypoint-initdb.d/002_roles_and_permissions.sql
	$(COMPOSE) exec postgres psql -U churn_admin -d churn_db \
		-f /docker-entrypoint-initdb.d/003_data_quality.sql
	@echo "All migrations applied."

seed:
	$(COMPOSE) exec postgres psql -U churn_admin -d churn_db \
		-f /dev/stdin < db/seeds/sample_data.sql

validate-schema:
	@echo "Checking for quarantine rows..."
	$(COMPOSE) exec postgres psql -U churn_admin -d churn_db \
		-c "SELECT failure_category, COUNT(*) FROM raw.quarantine WHERE resolved=FALSE GROUP BY 1 ORDER BY 2 DESC;"

# ── Testing ─────────────────────────────────────────────────
test:
	PYTHONPATH=apps/api:ml:pipelines pytest tests/ -v --tb=short

test-unit:
	PYTHONPATH=apps/api:ml:pipelines pytest tests/unit/ -v

test-integration:
	PYTHONPATH=apps/api:ml:pipelines pytest tests/integration/ -v --tb=short

test-leakage:
	@echo "Running leakage gate — must pass 100%..."
	PYTHONPATH=apps/api:ml:pipelines pytest tests/unit/test_leakage.py -v --tb=long
	@echo "✅ Leakage gate passed"

test-smoke:
	BASE_URL=http://localhost:8000 \
	API_SECRET_KEY=$$(grep API_SECRET_KEY $(ENV_FILE) | cut -d= -f2) \
	PYTHONPATH=apps/api pytest tests/smoke/ -v

# ── Code quality ────────────────────────────────────────────
lint:
	ruff check . --select=E,F,W,I

format:
	black .
	isort .

type-check:
	mypy apps/api --ignore-missing-imports

# ── Ingestion ─────────────────────────────────────────────── 
SOURCE_CUSTOMERS ?= data/customers.csv
SOURCE_BILLING   ?= data/billing.csv
SOURCE_SUPPORT   ?= data/support.csv

ingest-customers:
	$(COMPOSE) exec training_worker python /app/pipelines/ingestion/ingest_customers.py \
		--source-file /data/customers.csv --source-system csv

ingest-billing:
	$(COMPOSE) exec training_worker python /app/pipelines/ingestion/ingest_billing.py \
		--source-file /data/billing.csv --source-system csv

ingest-support:
	$(COMPOSE) exec training_worker python /app/pipelines/ingestion/ingest_support.py \
		--source-file /data/support.csv --source-system csv

# ── ML Pipelines ────────────────────────────────────────────
TODAY = $(shell date +%Y-%m-%d)

feature-snapshot:
	$(COMPOSE) exec training_worker python /app/ml/features/feature_store.py \
		--observation-date $(TODAY) \
		--horizon-days 30 \
		--feature-version v1.0.0

label-data:
	$(COMPOSE) exec training_worker python /app/pipelines/labeling_pipeline.py \
		--observation-date $(TODAY) \
		--horizon-days 30

score-batch:
	$(COMPOSE) exec training_worker python /app/pipelines/scoring_pipeline.py \
		--observation-date $(TODAY)

monitor:
	YESTERDAY=$$(date -d "yesterday" +%Y-%m-%d 2>/dev/null || date -v -1d +%Y-%m-%d); \
	$(COMPOSE) exec training_worker python /app/pipelines/monitoring_pipeline.py \
		--reference-date $$YESTERDAY \
		--monitoring-date $(TODAY)

# ── Training ─────────────────────────────────────────────────
train:
	$(COMPOSE) exec training_worker python /app/pipelines/training_pipeline.py \
		--observation-date $(TODAY) \
		--horizon-days 30 \
		--feature-version v1.0.0

train-tuned:
	$(COMPOSE) exec training_worker python /app/pipelines/training_pipeline.py \
		--observation-date $(TODAY) \
		--horizon-days 30 \
		--feature-version v1.0.0 \
		--run-tuning

# ── MLOps ────────────────────────────────────────────────────
check-triggers:
	YESTERDAY=$$(date -d "yesterday" +%Y-%m-%d 2>/dev/null || date -v -1d +%Y-%m-%d); \
	$(COMPOSE) exec training_worker python /app/pipelines/mlops/retraining_trigger.py \
		--reference-date $$YESTERDAY \
		--monitoring-date $(TODAY)

promote:
	@echo "Usage: make promote CANDIDATE_VERSION=<version> CANDIDATE_RUN_ID=<run_id>"
	$(COMPOSE) exec training_worker python /app/pipelines/mlops/promotion_gate.py \
		--candidate-run-id $(CANDIDATE_RUN_ID) \
		--candidate-version $(CANDIDATE_VERSION) \
		--feature-version v1.0.0 \
		--observation-date $(TODAY)

rollback:
	$(COMPOSE) exec training_worker python /app/pipelines/mlops/rollback.py --dry-run

rollback-execute:
	@echo "Executing LIVE rollback — confirm? [y/N]"
	@read CONFIRM && [ "$$CONFIRM" = "y" ] && \
		$(COMPOSE) exec training_worker python /app/pipelines/mlops/rollback.py \
		|| echo "Rollback cancelled."

# ── Metrics ──────────────────────────────────────────────────
metrics:
	curl -s -H "Authorization: Bearer $$(grep SECRET_KEY $(ENV_FILE) | cut -d= -f2)" \
		http://localhost:8000/metrics/json | python -m json.tool

# ── Cleanup ─────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
