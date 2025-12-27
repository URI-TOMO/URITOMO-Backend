"""
FastAPI Application Entry Point
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.errors import (
    AppError,
    ValidationError,
    app_exception_handler,
    http_exception_handler,
    validation_exception_handler,
    general_exception_handler,
)
from app.core.logging import setup_logging, RequestIDMiddleware
from app.infra.db import close_db_connection
from app.infra.redis import init_redis_pool, close_redis_pool
from app.infra.qdrant import init_qdrant_client, close_qdrant_client, ensure_collections_exist
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging()
    await init_redis_pool()
    await init_qdrant_client()
    
    # Initialize Qdrant collections background text
    # In production, this might be better as a migration step
    try:
        await ensure_collections_exist()
    except Exception as e:
        # Don't fail startup if qdrant is down, but log it
        print(f"Warning: Failed to initialize Qdrant collections: {e}")
        
    yield
    
    # Shutdown
    await close_redis_pool()
    await close_qdrant_client()
    await close_db_connection()


def create_app() -> FastAPI:
    app = FastAPI(
        title="URITOMO Backend",
        description="Real-time translation and cultural context service",
        version="0.1.0",
        openapi_url=f"{settings.api_prefix}/openapi.json",
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_credentials,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
    )

    # Routes
    app.include_router(api_router, prefix=settings.api_prefix)

    # Exception Handlers
    app.add_exception_handler(AppError, app_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    return app


app = create_app()
