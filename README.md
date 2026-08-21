# Customer Churn Prediction & Retention Intelligence Platform

> **Decision-support system** — provides churn probability scores and SHAP-driven explanations to retention analysts. No autonomous retention actions.

---

## Architecture at a Glance

```
Data Sources → Ingestion → PostgreSQL (raw/curated/ml) → Feature Snapshots
                                                               ↓
                                                     ML Training (XGBoost/LightGBM)
                                                               ↓
                                                     MLflow Model Registry (champion)
                                                               ↓
                                FastAPI ←───────── Champion Model (loaded on startup)
                                    ↑                          ↓
                             Retention Analyst         ml.predictions (SHAP drivers)
                             (via API / Dashboard)              ↓
                                                        Apache Superset Dashboard
```

---

## Quick Start (Windows PowerShell)

### Prerequisites
- Windows 10/11 with PowerShell
- Docker Desktop ≥ 4.x (with Docker Compose plugin)
- Python ≥ 3.10

### 1. Configure Environment

```powershell
Copy-Item .env.example .env
# Edit .env if needed (default development keys are pre-generated)
```

### 2. Start All Services

```powershell
docker compose up -d
```

| Service | URL |
|---|---|
| FastAPI (Prediction API) | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/api/docs |
| ReDoc UI | http://localhost:8000/api/redoc |
| MLflow UI | http://localhost:5000 |
| Superset Dashboard | http://localhost:8088 |

### 3. Run Database Migrations & Seed Data

```powershell
# Run schema and role migrations
docker compose exec postgres psql -U churn_admin -d churn_db -f /docker-entrypoint-initdb.d/001_initial_schema.sql
docker compose exec postgres psql -U churn_admin -d churn_db -f /docker-entrypoint-initdb.d/002_roles_and_permissions.sql
docker compose exec postgres psql -U churn_admin -d churn_db -f /docker-entrypoint-initdb.d/003_data_quality.sql

# Seed initial sample data
Get-Content db/seeds/sample_data.sql | docker compose exec -T postgres psql -U churn_admin -d churn_db
```

### 4. Run Test Suite

```powershell
# Run unit tests
python -m pytest tests/unit/ -v

# Run leakage prevention gate (must pass 100%)
python -m pytest tests/unit/test_leakage.py -v

# Run integration tests
python -m pytest tests/integration/ -v
```

---

## ML Pipeline

### Step-by-Step Model Training

```powershell
# 1. Start background training worker
docker compose --profile training up -d

# 2. Extract point-in-time feature snapshots
docker compose exec training_worker python /app/ml/features/feature_store.py --observation-date 2024-01-01 --horizon-days 30 --feature-version v1.0.0

# 3. Generate churn ground-truth labels
docker compose exec training_worker python /app/pipelines/labeling_pipeline.py --observation-date 2024-01-01 --horizon-days 30

# 4. Train candidate ML models (Logistic Regression, LightGBM, XGBoost, CatBoost)
docker compose exec training_worker python /app/pipelines/training_pipeline.py --observation-date 2024-01-01 --horizon-days 30 --feature-version v1.0.0

# 5. Evaluate promotion gates
docker compose exec training_worker python /app/pipelines/mlops/promotion_gate.py --candidate-run-id <run_id> --candidate-version 1.0.0 --observation-date 2024-01-01
```

### Daily Batch Scoring & Monitoring

```powershell
# Score all active customers → ml.predictions table
docker compose exec training_worker python /app/pipelines/scoring_pipeline.py --observation-date 2024-01-01

# Calculate PSI drift metrics & calibration checks
docker compose exec training_worker python /app/pipelines/monitoring_pipeline.py --reference-date 2023-12-31 --monitoring-date 2024-01-01
```

---

## API Reference

Full Interactive OpenAPI Specification: http://localhost:8000/api/docs

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/ready` | Readiness check (DB & Model loaded) |
| GET | `/v1/customers/{id}/risk` | Retrieve risk score & SHAP drivers |
| POST | `/v1/predict` | Single customer prediction |
| POST | `/v1/predict/batch` | Batch predictions (≤1000 items) |
| GET | `/v1/models/current` | Active champion model metadata |

**Authentication:** Bearer Token (`Authorization: Bearer <API_SECRET_KEY>`)

---

## Repository Structure

```
.
├── apps/api/               FastAPI prediction service & auth middleware
├── apps/dashboard/         Superset config & dashboard definitions
├── db/migrations/          PostgreSQL DDL migrations (001–003)
├── db/seeds/               Sample data for local dev
├── ml/
│   ├── features/           Feature config, SQL queries, feature store
│   ├── training/           Trainer, tuner, splitter, model configs
│   ├── evaluation/         Evaluator, calibration, threshold selector
│   └── explainability/     SHAP explainer & business labels
├── pipelines/              Ingestion, labeling, scoring, monitoring
├── tests/
│   ├── unit/               Leakage, features, schemas, auth tests
│   ├── integration/        API + pipeline end-to-end integration
│   ├── load/               Locust SLA throughput tests
│   └── smoke/              Post-deployment smoke tests
├── infra/
│   ├── docker-compose.yml  Docker Compose configuration
│   ├── nginx/              Reverse proxy config
│   └── aws/                EC2 bootstrap, IAM policies
└── .github/workflows/      CI/CD pipelines & promotion workflows
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Primary Metric | PR-AUC | Optimizes for precision/recall under class imbalance |
| Operating Point | Retention Capacity | Frozen threshold based on operational budget |
| Feature Store | PostgreSQL `ml.feature_snapshots` | Point-in-time safe; no external infra needed |
| Explainability | SHAP `TreeExplainer` | Fast local/global feature attributions |
| Calibration | Isotonic / Sigmoid (Validation Hold-out) | Well-calibrated probabilities on check set |
| Test Set Policy | Single Evaluation | Prevents subtle test-set leakage |

---

## Security & Compliance

- **Least-Privilege DB Roles:** `api_user`, `ml_user`, `ingestion_user`, `dashboard_user`
- **PII Minimisation:** Email and phone stored as SHA-256 hashes
- **Constant-time Auth:** Bearer token comparison via `hmac.compare_digest`
- **IP Anonymisation:** Client IPs masked in structured logs (GDPR compliant)
- **Secrets Management:** `.env` for local dev; AWS Secrets Manager for production

---

## License

MIT
