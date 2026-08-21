"""
apps/dashboard/superset_config.py
Apache Superset configuration for the Churn Intelligence Platform.

Environment variables required:
  SUPERSET_SECRET_KEY          : Random string for session encryption
  DATABASE_URL                 : PostgreSQL DSN for churn_db
  SUPERSET_ADMIN_USERNAME      : Admin username
  SUPERSET_ADMIN_EMAIL         : Admin email
  SUPERSET_ADMIN_PASSWORD      : Admin password

Mount this file to /app/pythonpath/superset_config.py inside the Superset container.
"""
from __future__ import annotations

import os

# ── Core ─────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SUPERSET_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")

# ── Database ──────────────────────────────────────────────────
SQLALCHEMY_DATABASE_URI = os.getenv(
    "SUPERSET_META_DB",
    "postgresql+psycopg2://superset:superset@postgres:5432/superset_meta",
)

# ── Feature flags ─────────────────────────────────────────────
FEATURE_FLAGS: dict = {
    "ENABLE_TEMPLATE_PROCESSING":        True,   # Allow Jinja in SQL
    "ALERT_REPORTS":                     True,   # Email/Slack alerts
    "DASHBOARD_NATIVE_FILTERS":          True,
    "DASHBOARD_CROSS_FILTERS":           True,
    "DRILL_TO_DETAIL":                   True,
    "ENABLE_ADVANCED_DATA_TYPES":        True,
}

# ── Cache (use Redis in production) ──────────────────────────
CACHE_CONFIG = {
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 300,
}

# ── Security ──────────────────────────────────────────────────
WTF_CSRF_ENABLED    = True
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"

# ── Row-level security ────────────────────────────────────────
# Retention analysts see only predictions; ML engineers see everything.
# Configure via Superset UI → Security → Row Level Security.

# ── Allowed hosts ─────────────────────────────────────────────
ALLOWED_HOSTS = ["*"]   # Restrict in production

# ── Logging ──────────────────────────────────────────────────
import logging
logging.basicConfig(level=logging.INFO)
