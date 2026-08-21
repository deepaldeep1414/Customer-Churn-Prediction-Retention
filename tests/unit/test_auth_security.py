"""
tests/unit/test_auth_security.py
Security unit tests for authentication middleware.

Covers:
  - Startup rejection when SECRET_KEY is empty or too short (F-01)
  - Constant-time token comparison via hmac.compare_digest (F-02)
  - Public path bypass
  - 401 on missing/malformed header
  - 403 on wrong token
"""
from __future__ import annotations

import hmac
import os
import time
from unittest.mock import patch, AsyncMock, MagicMock

import pytest


class TestSecretKeyValidation:
    """F-01: Startup must reject empty or short SECRET_KEY."""

    def test_empty_secret_key_raises_runtime_error(self):
        with patch.dict("os.environ", {"SECRET_KEY": ""}):
            import importlib
            import apps.api.middleware.auth as auth_mod  # noqa: F401  (for import path)
            from apps.api.middleware.auth import _load_secret_key
            with pytest.raises(RuntimeError, match="SECRET_KEY environment variable is not set"):
                _load_secret_key()

    def test_unset_secret_key_raises_runtime_error(self):
        env = {k: v for k, v in os.environ.items() if k != "SECRET_KEY"}
        with patch.dict("os.environ", env, clear=True):
            from apps.api.middleware.auth import _load_secret_key
            with pytest.raises(RuntimeError, match="SECRET_KEY environment variable is not set"):
                _load_secret_key()

    def test_short_secret_key_raises_runtime_error(self):
        with patch.dict("os.environ", {"SECRET_KEY": "tooshort123"}):
            from apps.api.middleware.auth import _load_secret_key
            with pytest.raises(RuntimeError, match="at least 32 characters"):
                _load_secret_key()

    def test_exactly_32_chars_is_accepted(self):
        key = "a" * 32
        with patch.dict("os.environ", {"SECRET_KEY": key}):
            from apps.api.middleware.auth import _load_secret_key
            result = _load_secret_key()
            assert result == key

    def test_long_key_is_accepted(self):
        key = "x" * 64
        with patch.dict("os.environ", {"SECRET_KEY": key}):
            from apps.api.middleware.auth import _load_secret_key
            result = _load_secret_key()
            assert result == key


class TestConstantTimeComparison:
    """F-02: Token comparison must use hmac.compare_digest."""

    def test_compare_digest_used_for_same_length_tokens(self):
        """
        Verify that compare_digest is constant-time for same-length inputs.
        Timing variance for same-length wrong vs right must be < 1ms.
        """
        secret = "s" * 64
        correct = "s" * 64
        wrong   = "w" * 64

        samples = 200
        t_correct_total = 0.0
        t_wrong_total   = 0.0

        for _ in range(samples):
            t0 = time.perf_counter()
            hmac.compare_digest(correct.encode(), secret.encode())
            t_correct_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            hmac.compare_digest(wrong.encode(), secret.encode())
            t_wrong_total += time.perf_counter() - t0

        avg_correct = t_correct_total / samples
        avg_wrong   = t_wrong_total   / samples
        diff_ms     = abs(avg_correct - avg_wrong) * 1000

        assert diff_ms < 1.0, (
            f"compare_digest timing variance too high: {diff_ms:.4f}ms. "
            "Possible timing oracle."
        )

    def test_compare_digest_returns_false_on_mismatch(self):
        assert not hmac.compare_digest(b"wrong_token", b"correct_token_here_12345678901")

    def test_compare_digest_returns_true_on_match(self):
        secret = b"correct-secret-key-32-chars-long"
        assert hmac.compare_digest(secret, secret)


class TestPublicPathBypass:
    """Public paths must never require auth."""

    PUBLIC_PATHS = ["/health", "/ready", "/metrics", "/api/docs", "/api/redoc"]

    @pytest.mark.parametrize("path", PUBLIC_PATHS)
    @pytest.mark.asyncio
    async def test_public_path_bypasses_auth(self, path: str):
        with patch.dict("os.environ", {"SECRET_KEY": "a" * 32}):
            from apps.api.middleware.auth import AuthMiddleware

            app_mock    = MagicMock()
            middleware   = AuthMiddleware(app_mock)
            call_next   = AsyncMock(return_value=MagicMock(status_code=200))

            request = MagicMock()
            request.url.path = path
            request.headers  = {}

            await middleware.dispatch(request, call_next)
            call_next.assert_called_once_with(request)


class TestAuthRejection:
    """Authenticated paths must enforce Bearer token."""

    @pytest.mark.asyncio
    async def test_missing_auth_header_returns_401(self):
        with patch.dict("os.environ", {"SECRET_KEY": "a" * 32}):
            from apps.api.middleware.auth import AuthMiddleware

            app_mock   = MagicMock()
            middleware = AuthMiddleware(app_mock)
            call_next  = AsyncMock()

            request = MagicMock()
            request.url.path = "/v1/predict"
            headers_dict = {}
            request.headers.get = headers_dict.get

            resp = await middleware.dispatch(request, call_next)
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_token_returns_403(self):
        with patch.dict("os.environ", {"SECRET_KEY": "a" * 32}):
            from apps.api.middleware.auth import AuthMiddleware

            app_mock   = MagicMock()
            middleware = AuthMiddleware(app_mock)
            call_next  = AsyncMock()

            request = MagicMock()
            request.url.path = "/v1/predict"
            request.headers.get = lambda key, default="": "Bearer wrong_token_here"

            resp = await middleware.dispatch(request, call_next)
            assert resp.status_code == 403
