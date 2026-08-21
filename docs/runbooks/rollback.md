# Rollback Runbook
## Customer Churn Platform — Champion Model Rollback

> **Use this runbook when** a post-deployment smoke test fails, monitoring alerts fire, or the retention team reports unexpected scores.

---

## Step 1 — Identify previous champion

```sql
-- Run in psql (churn_admin)
SELECT model_version, algorithm, pr_auc, promoted_at, promoted_by, stage
FROM ml.model_registry
WHERE stage IN ('champion', 'archived')
ORDER BY promoted_at DESC
LIMIT 5;
```

Note the `model_version` of the previous champion (the first `archived` row).

---

## Step 2 — Archive current champion

```sql
UPDATE ml.model_registry
SET stage       = 'archived',
    archived_at = now()
WHERE stage = 'champion';
```

---

## Step 3 — Re-promote previous version

```sql
-- Replace <PREVIOUS_VERSION> with the version from Step 1
UPDATE ml.model_registry
SET stage        = 'champion',
    promoted_at  = now(),
    promoted_by  = 'rollback',
    promotion_notes = 'Emergency rollback — see incident ticket'
WHERE model_version = '<PREVIOUS_VERSION>';
```

---

## Step 4 — Rollback in MLflow registry

```python
import mlflow
client = mlflow.tracking.MlflowClient()
client.set_tracking_uri("http://mlflow:5000")

# Archive current Production
for mv in client.get_latest_versions("churn_predictor", stages=["Production"]):
    client.transition_model_version_stage("churn_predictor", mv.version, "Archived")

# Restore previous version
client.transition_model_version_stage("churn_predictor", "<PREVIOUS_VERSION>", "Production")
```

---

## Step 5 — Restart API (picks up new champion on load)

```bash
# Local
docker compose -f infra/docker-compose.yml restart api

# Production (SSH to EC2)
cd /opt/churn && docker compose -f docker-compose.prod.yml restart api
```

---

## Step 6 — Verify rollback

```bash
# Check which model is loaded
curl -s -H "Authorization: Bearer $API_SECRET_KEY" \
  http://localhost:8000/v1/models/current | python -m json.tool

# Run smoke tests
pytest tests/smoke/ -v
```

---

## Step 7 — Post-incident actions

- [ ] Open incident ticket referencing the bad `model_version`
- [ ] Block bad run from re-promotion (add `promotion_notes` with "BLOCKED")
- [ ] Root cause analysis: check training logs, data quality, feature drift
- [ ] Re-run full evaluation before next promotion attempt
