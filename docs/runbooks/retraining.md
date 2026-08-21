# Retraining Runbook
## Customer Churn Platform — Model Retraining Guide

> **When to use this runbook**: Monthly scheduled retraining, after a PSI drift alert, or when ground-truth label volume meets the threshold.

---

## Prerequisites

- Access to the `model-promotion` GitHub Actions environment (ML engineer role)
- `DATABASE_URL`, `MLFLOW_URI` environment variables
- SSH access to EC2 instance (or VPN to internal Kubernetes cluster)

---

## Automated Path (Recommended)

### Trigger monthly retraining

```bash
# Via GitHub Actions (preferred — full audit trail)
gh workflow run retrain.yml --field force_retrain=false

# With Optuna tuning (slower, ~4h — do quarterly)
gh workflow run retrain.yml --field run_tuning=true
```

The workflow runs:
1. **Trigger check** → evaluates PSI, PR-AUC decay, label volume
2. **Training** → all 4 algorithms with feature version v1.0.0
3. **7 promotion gates** (automated — fails loudly if any gate fails)
4. **Human approval** → ML engineer approves in GitHub Environments
5. **Promotion** → MLflow + DB champion update + deployment trigger

---

## Manual Path (Debugging / Emergency)

### Step 1 — Build feature snapshots

```bash
# On EC2 (or local with DATABASE_URL set)
cd /opt/churn
docker compose --profile training exec training_worker \
  python ml/features/feature_store.py \
    --observation-date 2024-01-31 \
    --horizon-days 30 \
    --feature-version v1.0.0
```

### Step 2 — Generate labels

```bash
docker compose --profile training exec training_worker \
  python pipelines/labeling_pipeline.py \
    --observation-date 2024-01-31 \
    --horizon-days 30 \
    --label-version 1
```

### Step 3 — Run leakage audit (MANDATORY before training)

```sql
-- Must return 0 rows
SELECT COUNT(*) FROM ml.validate_snapshot_leakage('v1.0.0', '2024-01-31'::DATE);
```

If this returns > 0, **DO NOT TRAIN**. Fix the feature SQL first.

### Step 4 — Run training pipeline

```bash
docker compose --profile training exec training_worker \
  python pipelines/training_pipeline.py \
    --observation-date 2024-01-31 \
    --horizon-days 30 \
    --feature-version v1.0.0
```

Watch for output like:
```
Champion selected: algorithm=xgboost val_pr_auc=0.8312
candidate_version: 20240131.xgboost
```

### Step 5 — Run promotion gates

```bash
docker compose --profile training exec training_worker \
  python pipelines/mlops/promotion_gate.py \
    --candidate-run-id <RUN_ID_FROM_STEP_4> \
    --candidate-version 20240131.xgboost \
    --feature-version v1.0.0 \
    --observation-date 2024-01-31
```

All 7 gates must show ✅. If any show ❌, do NOT promote.

### Step 6 — Promote (after human review)

```bash
python pipelines/mlops/promotion_gate.py \
  --candidate-version 20240131.xgboost \
  # Or use the GitHub Actions model-promotion workflow for full audit trail
```

### Step 7 — Deploy & verify

```bash
# Trigger deployment workflow
gh workflow run deploy.yml --field image_tag=latest

# Or manually on EC2
cd /opt/churn
docker compose -f docker-compose.prod.yml up -d --no-deps api
sleep 15
curl -s -H "Authorization: Bearer $SECRET_KEY" \
  http://localhost:8000/v1/models/current | python -m json.tool
```

---

## Rollback

See [rollback.md](./rollback.md).

---

## Data Quality Checks During Retraining

```sql
-- Check quarantine rate for recent batches
SELECT source_table, 
       SUM(quarantined_records) AS quarantined,
       ROUND(SUM(quarantined_records)::NUMERIC / NULLIF(SUM(total_records),0)*100, 2) AS rate_pct
FROM raw.quality_audit_log
WHERE validated_at >= now() - INTERVAL '7 days'
GROUP BY source_table;

-- Any quarantine rate > 5% should block retraining
```

---

## Key Metrics to Monitor Post-Training

| Metric | Tool | Acceptable Range |
|--------|------|-----------------|
| val_pr_auc | MLflow | ≥ 0.75 |
| val_ece | MLflow | ≤ 0.05 |
| test_pr_auc | MLflow | ≥ 0.73 |
| Predicted churn rate | Superset | Within ±5% of last month |
| Quarantine rate | DB | < 5% per source table |
