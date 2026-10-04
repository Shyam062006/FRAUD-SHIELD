"""Application configuration and settings for FRAUD SHIELD backend."""

import os
from pathlib import Path
from typing import List, Set
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_existing_path(relative_or_abs: str) -> Path:
    """Resolve a file path by checking common roots (backend/, workspace root, or absolute)."""
    candidate = Path(relative_or_abs)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    # Try relative to current working directory
    if candidate.exists():
        return candidate.resolve()

    # Try relative to backend directory
    backend_dir = Path(__file__).resolve().parent.parent
    if (backend_dir / relative_or_abs).exists():
        return (backend_dir / relative_or_abs).resolve()

    # Try relative to workspace root (parent of backend)
    workspace_dir = backend_dir.parent
    if (workspace_dir / relative_or_abs).exists():
        return (workspace_dir / relative_or_abs).resolve()

    # Default to resolved relative to backend directory if not found yet
    return (backend_dir / relative_or_abs).resolve()


class Settings(BaseSettings):
    """Application settings with environment variable overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server Settings
    APP_NAME: str = "FRAUD SHIELD"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Artifact Paths
    MODEL_PATH: str = "../ML/model.pkl"
    SCALER_PATH: str = "../ML/scaler.pkl"
    METRICS_PATH: str = "app/metrics.json"
    FRONTEND_HTML_PATH: str = "../frontend/index.html"

    # Decision Threshold (notebook: 0.50, UI visualizer: 0.55)
    DECISION_THRESHOLD: float = Field(default=0.50, ge=0.0, le=1.0)

    # UI Validation Boundaries
    MIN_AMOUNT: float = 1.0
    MAX_AMOUNT: float = 1000000.0
    MIN_BALANCE: float = 0.0
    MAX_BALANCE: float = 10000000.0
    ALLOWED_TYPES: Set[str] = {"TRANSFER", "CASH_OUT"}

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 120

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    @property
    def resolved_model_path(self) -> Path:
        return _resolve_existing_path(self.MODEL_PATH)

    @property
    def resolved_scaler_path(self) -> Path:
        return _resolve_existing_path(self.SCALER_PATH)

    @property
    def resolved_metrics_path(self) -> Path:
        return _resolve_existing_path(self.METRICS_PATH)

    @property
    def resolved_frontend_html_path(self) -> Path:
        return _resolve_existing_path(self.FRONTEND_HTML_PATH)


settings = Settings()
