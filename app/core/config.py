"""
URITOMO Backend Core Configuration

This module manages all application configuration using Pydantic Settings.
Environment variables are loaded from .env file.
"""

from typing import Literal, Optional
from functools import lru_cache

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"

    api_prefix: str = ""

    # Database
    database_url: str = Field(
        default="mysql+aiomysql://uritomo_user:uritomo_pass@localhost:3306/uritomo"
    )
    db_echo: bool = False
    db_pool_size: int = 20
    db_max_overflow: int = 40

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_db: int = 0
    # Security
    jwt_secret_key: str = Field(min_length=32)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_minutes: int = 30

    # External APIs
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-5-nano"

    deepl_api_key: Optional[str] = None

    # LiveKit
    livekit_url: Optional[str] = None
    livekit_api_key: Optional[str] = None
    livekit_api_secret: Optional[str] = None

    # Translation Settings
    translation_provider: Literal["OPENAI", "DEEPL", "MOCK"] = "MOCK"

    # Background Worker
    worker_service_key: Optional[str] = None

    # CORS
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:8000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ]
    cors_credentials: bool = True
    cors_methods: list[str] = ["*"]
    cors_headers: list[str] = ["*"]

    # Feature Flags
    enable_websocket: bool = True

    @validator("cors_origins", pre=True)
    def assemble_cors_origins(cls, v: any) -> list[str]:
        """Parse CORS origins from string, list, or JSON string"""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            import json
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    return [v]
            return v
        return ["*"]

    @validator("jwt_secret_key")
    def validate_jwt_secret(cls, v: str) -> str:
        """Ensure JWT secret is strong enough"""
        if len(v) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters long")
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.env == "production"

    @property
    def use_mock_translation(self) -> bool:
        """Check if using mock translation"""
        return self.translation_provider == "MOCK" or not self.openai_api_key

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache for singleton pattern.
    """
    return Settings()


# Export singleton instance
settings = get_settings()
