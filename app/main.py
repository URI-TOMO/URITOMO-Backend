"""
FastAPI Application Entry Point
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer

from app.api.router import api_router

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


tags_metadata = [
    {
        "name": "debug",
        "description": "Debug tools for seeding data and testing.",
    },
    {
        "name": "Example Token Auth",
        "description": "Tokens for testing authentication. Get your debug token here!",
    },
    {
        "name": "Example CRUD",
        "description": "Quickly setup users and mock data for development.",
    },
    {
        "name": "auth",
        "description": "Standard Authentication operations (Login, Register).",
    },

    {
        "name": "orgs",
        "description": "Manage Organizations and their glossaries.",
    },
    {
        "name": "meetings",
        "description": "Create and manage translation meetings.",
    },
    {
        "name": "segments",
        "description": "Ingest and process transcript segments.",
    },
    {
        "name": "websocket",
        "description": "Real-time communication using WebSocket.",
    },
    {
        "name": "health",
        "description": "System health check.",
    },
]


def create_app() -> FastAPI:
    app = FastAPI(
        title="URITOMO Backend",
        description="""
URITOMO API provides real-time translation with cultural context explanations.

## Features
* **Real-time Translation**: Low-latency WebSocket streaming.
* **Cultural Context**: RAG-powered explanations for nuanced expressions.
* **Meeting Management**: Complete lifecycle for multilingual sessions.
* **Organizations**: Shared glossaries and cultural knowledge.
""",
        version="0.1.0",
        openapi_tags=tags_metadata,
        openapi_url=f"{settings.api_prefix}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        swagger_ui_parameters={
            "persistAuthorization": True,
            "displayRequestDuration": True
        },
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

    # Root Route
    @app.get("/", tags=["health"], include_in_schema=False)
    async def root():
        return {
            "message": "Welcome to URITOMO Backend API",
            "docs": "/docs",
            "status": "operational"
        }


    # Routes

    # We include dependencies=[Depends(HTTPBearer())] if we want to FORCE it everywhere globally.
    # But usually, it's better to apply it to the main api_router.
    # Routes
    app.include_router(api_router, prefix=settings.api_prefix)

    # Exception Handlers
    app.add_exception_handler(AppError, app_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    return app


app = create_app()
