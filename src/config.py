"""Centralized configuration: .env settings + config.yaml loader."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment variables loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenRouter
    openrouter_api_key: str
    openrouter_model: str = "z-ai/glm-4.5-air:free"

    # Environment
    log_level: str = "INFO"


def load_yaml_config(path: Optional[Path] = None) -> dict:
    """Load config.yaml from project root."""
    if path is None:
        path = Path(__file__).parent.parent / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
