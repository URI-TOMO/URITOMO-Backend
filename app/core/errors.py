"""
Error handling configuration

Custom exception classes and exception handlers for FastAPI.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.core.logging import get_logger

logger = get_logger(__name__)


class AppError(Exception):
    """Base exception class for application errors"""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


class NotFoundError(AppError):
    """Resource not found error"""

    def __init__(self, message: str = "Resource not found", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            code="NOT_FOUND",
            details=details,
        )


class ValidationError(AppError):
    """Data validation error"""

    def __init__(self, message: str = "Validation error", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="VALIDATION_ERROR",
            details=details,
        )


class AuthenticationError(AppError):
    """Authentication failed"""

    def __init__(self, message: str = "Authentication failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            code="AUTHENTICATION_FAILED",
            details=details,
        )


class PermissionError(AppError):
    """Permission denied"""

    def __init__(self, message: str = "Permission denied", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            code="PERMISSION_DENIED",
            details=details,
        )


async def app_exception_handler(request: Request, exc: AppError) -> JSONResponse:
    """Handle custom application exceptions"""
    logger.error(
        f"App error: {exc.message}",
        error_code=exc.code,
        status_code=exc.status_code,
        details=exc.details,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle standard FastAPI HTTP exceptions"""
    logger.warning(
        f"HTTP error: {exc.detail}",
        status_code=exc.status_code,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": str(exc.detail),
            }
        },
    )


async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Handle Pydantic validation errors"""
    errors = exc.errors()
    logger.warning(
        "Validation error",
        error_count=len(errors),
        path=request.url.path,
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {"errors": errors},
            }
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unhandled exceptions"""
    logger.exception(
        "Unhandled exception",
        error=str(exc),
        path=request.url.path,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
            }
        },
    )
