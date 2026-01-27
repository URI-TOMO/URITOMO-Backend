"""
Logging configuration for URITOMO Backend

Structured logging with request ID tracking and latency measurement.
"""

import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

import structlog
from pythonjsonlogger import jsonlogger

from app.core.config import settings

# Context variable for request ID tracking
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def add_request_id(logger: Any, method_name: str, event_dict: dict) -> dict:
    """Add request ID to log context"""
    request_id = request_id_var.get()
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


def setup_logging() -> None:
    """Configure structured logging for the application"""

    # Configure standard logging
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # JSON formatter for production
    if settings.is_production:
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s %(request_id)s"
        )
    else:
        # Human-readable format for development
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            add_request_id,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
            if settings.is_production
            else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    # Specific logger for WebSocket (Force visible on console)
    ws_logger = logging.getLogger("uritomo.ws")
    ws_logger.setLevel(logging.INFO)
    ws_handler = logging.StreamHandler(sys.stdout)
    ws_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    ws_logger.addHandler(ws_handler)
    ws_logger.propagate = False  # Prevent double logging


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a logger instance"""
    return structlog.get_logger(name)


class RequestIDMiddleware:
    """Middleware to track request IDs"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Generate request ID
        import uuid

        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)

        # Add to response headers
        async def send_with_request_id(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_request_id)


class RequestLoggingMiddleware:
    """Middleware to log every HTTP request/response"""

    def __init__(self, app):
        self.app = app
        self.logger = get_logger("app.request")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        status_code = None
        client = scope.get("client") or (None, None)
        query_string = scope.get("query_string", b"").decode("utf-8")

        self.logger.info(
            "request.start",
            method=scope.get("method"),
            path=scope.get("path"),
            query_string=query_string,
            client_host=client[0],
        )

        async def send_with_status(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status")
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        except Exception as exc:
            self.logger.exception(
                "request.error",
                method=scope.get("method"),
                path=scope.get("path"),
                query_string=query_string,
                client_host=client[0],
                error=str(exc),
            )
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(
                "request.end",
                method=scope.get("method"),
                path=scope.get("path"),
                query_string=query_string,
                status_code=status_code or 500,
                duration_ms=round(duration_ms, 2),
                client_host=client[0],
            )


class LatencyLogger:
    """Context manager for logging operation latency"""

    def __init__(self, operation: str, logger: structlog.stdlib.BoundLogger):
        self.operation = operation
        self.logger = logger
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.time() - self.start_time) * 1000
        self.logger.info(
            f"{self.operation} completed",
            operation=self.operation,
            latency_ms=round(latency_ms, 2),
            success=exc_type is None,
        )


        
