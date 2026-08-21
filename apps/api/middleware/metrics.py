"""
apps/api/middleware/metrics.py
Lightweight in-process latency metrics: p50/p95/p99 per endpoint.

Implements a sliding window (last N samples) using a circular buffer.
Exposes metrics via GET /metrics (Prometheus-compatible text format).

For production scale, swap this for prometheus_fastapi_instrumentator
or OpenTelemetry. This implementation has zero external dependencies
and is suitable for the v1 Docker Compose deployment.
"""
from __future__ import annotations

import collections
import json
import logging
import time
from typing import Callable, Deque

import numpy as np
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse

logger = logging.getLogger("churn.api.metrics")

# Rolling window size per route
WINDOW_SIZE = 1000

# Global storage: route → deque of (duration_ms, status_code) tuples
_latency_store: dict[str, Deque[tuple[float, int]]] = collections.defaultdict(
    lambda: collections.deque(maxlen=WINDOW_SIZE)
)
_error_counts:  dict[str, int] = collections.defaultdict(int)
_total_counts:  dict[str, int] = collections.defaultdict(int)


def _route_key(request: Request) -> str:
    """Normalise path to route key (strips path params)."""
    route = getattr(request, "scope", {}).get("route")
    if route:
        return f"{request.method} {route.path}"
    return f"{request.method} {request.url.path}"


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Records per-route latency samples and error counts.
    Thread-safe for single-process uvicorn deployments.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip the /metrics endpoint itself to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)

        start   = time.perf_counter()
        response: Response | None = None

        try:
            response = await call_next(request)
            return response
        except Exception:
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            status      = response.status_code if response else 500
            key         = _route_key(request)
            _latency_store[key].append((duration_ms, status))
            _total_counts[key] += 1
            if status >= 400:
                _error_counts[key] += 1


def compute_percentiles(samples: list[float], pcts: list[int]) -> dict[str, float]:
    if not samples:
        return {f"p{p}": 0.0 for p in pcts}
    arr = np.array(samples)
    return {f"p{p}": round(float(np.percentile(arr, p)), 2) for p in pcts}


def get_metrics_summary() -> dict:
    """Return structured metrics dict (JSON-serialisable)."""
    summary: dict[str, dict] = {}
    for route, window in _latency_store.items():
        durations = [d for d, _ in window]
        pcts = compute_percentiles(durations, [50, 95, 99])
        summary[route] = {
            **pcts,
            "p50_ms":      pcts["p50"],
            "p95_ms":      pcts["p95"],
            "p99_ms":      pcts["p99"],
            "total":       _total_counts[route],
            "errors":      _error_counts[route],
            "error_rate":  round(_error_counts[route] / max(_total_counts[route], 1), 4),
            "window_size": len(window),
        }
    return summary


def prometheus_metrics_text() -> str:
    """
    Render metrics in Prometheus exposition format.
    Suitable for scraping by Prometheus or Victoria Metrics.
    """
    lines: list[str] = []
    lines.append("# HELP churn_api_latency_ms Request latency in milliseconds")
    lines.append("# TYPE churn_api_latency_ms summary")

    for route, window in _latency_store.items():
        durations = [d for d, _ in window]
        if not durations:
            continue
        safe_route = route.replace(" ", "_").replace("/", "_").replace("{", "").replace("}", "")
        pcts = compute_percentiles(durations, [50, 95, 99])
        for pct_label, pct_val in pcts.items():
            quantile = str(int(pct_label[1:])/100)
            lines.append(
                f'churn_api_latency_ms{{route="{route}",quantile="{quantile}"}} {pct_val}'
            )
        lines.append(
            f'churn_api_requests_total{{route="{route}"}} {_total_counts[route]}'
        )
        lines.append(
            f'churn_api_errors_total{{route="{route}"}} {_error_counts[route]}'
        )

    return "\n".join(lines) + "\n"
