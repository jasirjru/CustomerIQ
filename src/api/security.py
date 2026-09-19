"""Bounded HTTP input, per-process rate limiting, security headers and safe logs."""

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import logging
import os
import time
import uuid

from starlette.responses import JSONResponse

from src.api.observability import canonical_route

logger = logging.getLogger("customeriq")
PROTECTED_PATHS = frozenset({"/metrics", "/model-info", "/predict", "/predict-batch"})


@dataclass(frozen=True)
class SecuritySettings:
    max_body_bytes: int = 262144
    requests_per_minute: int = 60
    allowed_origins: tuple[str, ...] = ()
    environment: str = "unspecified"
    require_api_key: bool = False
    api_keys: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self):
        normalized_environment = self.environment.strip().lower()
        if normalized_environment == "prod":
            normalized_environment = "production"
        object.__setattr__(self, "environment", normalized_environment)
        if self.max_body_bytes < 1024 or self.requests_per_minute < 1 or "*" in self.allowed_origins:
            raise ValueError("Use a positive rate, body limit >=1024 and explicit CORS origins.")
        if len(self.api_keys) > 32 or len(set(self.api_keys)) != len(self.api_keys):
            raise ValueError("Configure no more than 32 unique API keys.")
        if any(not key.isascii() or len(key) < 32 or len(key) > 256 for key in self.api_keys):
            raise ValueError("Each API key must contain 32 to 256 ASCII characters.")
        if self.require_api_key and not self.api_keys:
            raise ValueError("At least one API key is required when API-key enforcement is enabled.")
        if self.environment == "production" and not self.require_api_key:
            raise ValueError("Production must enforce API-key authentication.")

    @classmethod
    def from_env(cls):
        body = int(os.getenv("CUSTOMERIQ_MAX_BODY_BYTES", "262144"))
        rate = int(os.getenv("CUSTOMERIQ_RATE_LIMIT", "60"))
        origins = tuple(x.strip() for x in os.getenv("CUSTOMERIQ_CORS_ORIGINS", "").split(",") if x.strip())
        environment = os.getenv("CUSTOMERIQ_ENVIRONMENT", "unspecified").strip().lower()
        if environment == "prod":
            environment = "production"
        require_key_text = os.getenv(
            "CUSTOMERIQ_REQUIRE_API_KEY",
            "true" if environment.lower() == "production" else "false",
        ).strip().lower()
        if require_key_text not in {"true", "false"}:
            raise ValueError("CUSTOMERIQ_REQUIRE_API_KEY must be true or false.")
        require_key = require_key_text == "true"
        api_keys = tuple(x.strip() for x in os.getenv("CUSTOMERIQ_API_KEYS", "").split(",") if x.strip())
        return cls(
            max_body_bytes=body,
            requests_per_minute=rate,
            allowed_origins=origins,
            environment=environment,
            require_api_key=require_key,
            api_keys=api_keys,
        )


class SecurityMiddleware:
    def __init__(self, app, settings: SecuritySettings, metrics=None):
        self.app = app
        self.settings = settings
        self.metrics = metrics
        self.clients = OrderedDict()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        response_status = 500

        async def secured_send(message):
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = message["status"]
                csp = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                       "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
                       "frame-ancestors 'none'; form-action 'self'")
                message.setdefault("headers", []).extend([
                    (b"x-request-id", request_id.encode()),
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"cache-control", b"no-store"),
                    (b"content-security-policy", csp.encode()),
                    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                ])
            await send(message)

        async def reject(code, detail, headers=None):
            await JSONResponse({"detail": detail}, status_code=code, headers=headers)(scope, receive, secured_send)

        try:
            headers = dict(scope.get("headers", []))
            provided_key = headers.get(b"x-api-key", b"").decode("utf-8", errors="ignore")
            if self.settings.require_api_key and scope["path"] in PROTECTED_PATHS:
                if not any(hmac.compare_digest(provided_key, expected) for expected in self.settings.api_keys):
                    return await reject(401, "Authentication required.", {"WWW-Authenticate": "ApiKey"})
            if scope["path"] in ("/predict", "/predict-batch"):
                # Deliberately do not trust arbitrary X-Forwarded-For headers.
                client = (scope.get("client") or ("unknown",))[0]
                key = hashlib.sha256(provided_key.encode()).hexdigest()[:16] if provided_key else client
                now = time.monotonic()
                count, reset = self.clients.get(key, (0, now + 60))
                if now >= reset:
                    count, reset = 0, now + 60
                if count >= self.settings.requests_per_minute:
                    return await reject(429, "Request limit reached. Try again in one minute.")
                self.clients[key] = (count + 1, reset)
                self.clients.move_to_end(key)
                if len(self.clients) > 1024:
                    self.clients.popitem(last=False)
            if scope["method"] in ("POST", "PUT", "PATCH"):
                try:
                    length = int(headers.get(b"content-length", b"0"))
                except ValueError:
                    return await reject(400, "Invalid Content-Length.")
                if length < 0:
                    return await reject(400, "Invalid Content-Length.")
                if length > self.settings.max_body_bytes:
                    return await reject(413, "Request body exceeds the configured limit.")
                body = bytearray()
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    body.extend(message.get("body", b""))
                    if len(body) > self.settings.max_body_bytes:
                        return await reject(413, "Request body exceeds the configured limit.")
                    if not message.get("more_body", False):
                        break
                body_sent = False
                async def bounded_receive():
                    nonlocal body_sent
                    if body_sent:
                        return {"type": "http.request", "body": b"", "more_body": False}
                    body_sent = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                await self.app(scope, bounded_receive, secured_send)
            else:
                await self.app(scope, receive, secured_send)
        finally:
            route = canonical_route(scope)
            duration = time.monotonic() - started
            if self.metrics is not None:
                self.metrics.record_http(scope["method"], route, response_status, duration)
            logger.info(json.dumps({
                "event": "request", "request_id": request_id,
                "method": scope["method"], "route": route,
                "status": response_status, "duration_ms": round(duration * 1000, 2),
            }))
