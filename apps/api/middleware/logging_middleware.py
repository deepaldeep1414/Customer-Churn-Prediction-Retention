"""
apps/api/middleware/logging_middleware.py
Structured JSON logging for every HTTP request/response.

Logged fields per request:
  - request_id    : UUID generated per request (injected into response header X-Request-Id)
  - method        : HTTP verb
  - path          : URL path (no query params to avoid PII in logs)
  - status_code   : HTTP response status
  - duration_ms   : Total request duration in milliseconds
  - model_version : Champion model version (from app state)
  - client_ip     : Anonymised (last octet zeroed for IPv4)
  - error         : Error message if status >= 400

PII policy: query parameters and request bodies are NOT logged.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("churn.api.access")


import hashlib
import os

# Set LOG_CLIENT_IP=false to fully suppress client IP from logs (GDPR strict mode)
_LOG_CLIENT_IP = os.getenv("LOG_CLIENT_IP", "true").lower() not in ("false", "0", "no")


def _anonymise_ip(ip: str | None) -> str:
    """
    Anonymise client IP for structured logs.

    IPv4: zeroes the last two octets (e.g. 192.168.1.42 → 192.168.0.0)
          Per GDPR guidance, a /16 prefix is not considered personal data
          in most jurisdictions. The full /24 prefix can still be personal
          data (e.g. residential ISPs), so we zero two octets.

    IPv6: SHA-256 first 8 bytes (64-bit prefix), truncated to 16 hex chars.
          Full IPv6 addresses are unambiguously personal data.

    Env override: LOG_CLIENT_IP=false suppresses IP entirely.
    """
    if not _LOG_CLIENT_IP:
        return "suppressed"
    if not ip:
        return "unknown"

    # IPv4
    parts = ip.split(".")
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.0.0"

    # IPv6 — hash to irreversible token
    hashed = hashlib.sha256(ip.encode()).hexdigest()[:16]
    return f"ipv6:{hashed}"


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    Emits one structured JSON log line per request.
    Injects X-Request-Id into the response for end-to-end tracing.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id  = str(uuid.uuid4())
        start_time  = time.perf_counter()

        # Attach request_id to request state so handlers can log it
        request.state.request_id = request_id

        response: Response | None = None
        error_detail: str | None  = None

        try:
            response = await call_next(request)
        except Exception as exc:
            error_detail = str(exc)
            raise
        finally:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            # Model version from app state (may be None before load)
            loader       = getattr(request.app.state, "model_loader", None)
            model_version = loader.model_version if loader and loader.is_loaded else None

            log_record = {
                "event":         "http_request",
                "request_id":    request_id,
                "method":        request.method,
                "path":          request.url.path,
                "status_code":   response.status_code if response else 500,
                "duration_ms":   duration_ms,
                "model_version": model_version,
                "client_ip":     _anonymise_ip(request.client.host if request.client else None),
            }
            if error_detail:
                log_record["error"] = error_detail

            logger.info(json.dumps(log_record))

        # Inject request ID into response headers for client-side tracing
        if response is not None:
            response.headers["X-Request-Id"] = request_id
        return response
