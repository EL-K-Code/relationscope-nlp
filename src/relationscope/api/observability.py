"""Structured request logging without raw user text."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("relationscope.requests")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log metadata only: never request bodies, entity text, or sentences."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        started = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            payload = {
                "severity": "INFO" if status_code < 500 else "ERROR",
                "event": "http_request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
            logger.info(json.dumps(payload, separators=(",", ":")))
