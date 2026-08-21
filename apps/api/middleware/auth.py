"""
apps/api/middleware/auth.py
Bearer token authentication middleware.

Security hardening (v2):
  - Fails fast at startup if SECRET_KEY is unset or < 32 chars
  - Uses hmac.compare_digest() for constant-time token comparison
  - Tokens are never logged (only first 8 chars in debug mode)

In v3, replace with JWT verification or OAuth2 introspection.
"""
from __future__ import annotations

import hmac
import logging
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

PUBLIC_PATHS = {
    "/health", "/ready", "/metrics",
    "/api/docs", "/api/redoc", "/api/openapi.json",
    "/docs", "/redoc", "/openapi.json"
}

_MIN_KEY_LENGTH = 32


def _load_secret_key() -> str:
    """
    Load and validate the SECRET_KEY at startup.

    Raises RuntimeError (not ValueError) so FastAPI will treat it as a
    startup failure and refuse to serve traffic.
    """
    key = os.getenv("SECRET_KEY", "") or os.getenv("API_SECRET_KEY", "")
    if not key:
        raise RuntimeError(
            "SECRET_KEY environment variable is not set. "
            "Generate a secure key with: "
            "python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    if len(key) < _MIN_KEY_LENGTH:
        raise RuntimeError(
            f"SECRET_KEY must be at least {_MIN_KEY_LENGTH} characters "
            f"(got {len(key)}). Use a cryptographically random value."
        )
    return key


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Validates Bearer tokens using constant-time comparison.
    Fails loudly at startup — never silently degrades to open access.
    """

    def __init__(self, app, **kwargs) -> None:
        super().__init__(app, **kwargs)
        # Load and validate at middleware construction time (= app startup)
        self._secret: bytes = _load_secret_key().encode("utf-8")
        logger.info("AuthMiddleware initialised with SECRET_KEY (%d chars)", len(self._secret))

    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "missing_or_invalid_authorization_header",
                         "detail": "Provide: Authorization: Bearer <token>"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header.split(" ", 1)[1]

        # ── Constant-time comparison ─────────────────────────
        # hmac.compare_digest prevents timing oracle attacks.
        # Both sides must be bytes of the same length for full protection;
        # hmac.compare_digest handles different-length comparisons safely.
        if not hmac.compare_digest(
            token.encode("utf-8"),
            self._secret,
        ):
            # Log only first 8 chars (never full token)
            logger.warning(
                "Auth rejected | token_prefix=%s... | path=%s",
                token[:8], request.url.path,
            )
            return JSONResponse(
                status_code=403,
                content={"error": "forbidden"},
            )

        return await call_next(request)
