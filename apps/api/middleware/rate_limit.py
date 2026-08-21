"""
apps/api/middleware/rate_limit.py
Token-bucket rate limiter per Bearer token identity.
Prevents abuse of batch endpoints from a single client.

Limits (configurable via env):
  RATE_LIMIT_PER_MINUTE  : requests/minute per token (default: 120)
  RATE_LIMIT_BURST       : burst allowance (default: 20)
"""
from __future__ import annotations

import os
import time
import collections
import logging
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("churn.api.ratelimit")

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
RATE_LIMIT_BURST      = int(os.getenv("RATE_LIMIT_BURST",      "20"))

# {token_hash → (tokens_remaining, last_refill_time)}
_buckets: dict[str, list[float]] = {}

# Exempt paths (health/ready never rate-limited)
EXEMPT_PATHS = {"/health", "/ready", "/metrics", "/api/docs", "/api/redoc", "/api/openapi.json"}


def _get_token_key(request: Request) -> str:
    """Derive a stable non-PII key from the Bearer token."""
    auth = request.headers.get("Authorization", "")
    token = auth.split(" ", 1)[1] if auth.startswith("Bearer ") else "anonymous"
    # Use first 16 chars of token as bucket key (never log full token)
    return f"token:{token[:16]}"


def _check_rate_limit(key: str) -> tuple[bool, float]:
    """
    Token bucket check.
    Returns (allowed: bool, retry_after_seconds: float).
    """
    now        = time.monotonic()
    refill_rate = RATE_LIMIT_PER_MINUTE / 60.0  # tokens per second

    if key not in _buckets:
        _buckets[key] = [float(RATE_LIMIT_BURST), now]

    bucket = _buckets[key]
    tokens, last_refill = bucket

    # Refill tokens based on elapsed time
    elapsed = now - last_refill
    tokens  = min(RATE_LIMIT_BURST, tokens + elapsed * refill_rate)
    bucket[1] = now

    if tokens >= 1.0:
        bucket[0] = tokens - 1.0
        return True, 0.0
    else:
        bucket[0] = tokens
        retry_after = (1.0 - tokens) / refill_rate
        return False, round(retry_after, 1)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token-bucket rate limiter — applied after auth check."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        key     = _get_token_key(request)
        allowed, retry_after = _check_rate_limit(key)

        if not allowed:
            logger.warning(
                "Rate limit exceeded | key=%s... path=%s retry_after=%.1fs",
                key[:20], request.url.path, retry_after,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error":       "rate_limit_exceeded",
                    "detail":      f"Too many requests. Retry after {retry_after}s.",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)
