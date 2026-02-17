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
    openrouter_model: str = "google/gemma-3-27b-it:free"               # agent (vision)
    openrouter_backup_models: str = (                                   # fallback chain (comma-separated)
        "nvidia/nemotron-nano-12b-v2-vl:free,"
        "deepseek/deepseek-r1-0528:free,"
        "z-ai/glm-4.5-air:free,"
        "arcee-ai/trinity-mini:free,"
        "stepfun/step-3.5-flash:free"
    )
    openrouter_eval_model: str = "openrouter/aurora-alpha"              # evaluation (fast reasoning)

    @property
    def backup_models_list(self) -> list[str]:
        """Return backup models as a list, filtering out empty strings."""
        return [m.strip() for m in self.openrouter_backup_models.split(",") if m.strip()]

    # Environment
    log_level: str = "INFO"


def load_yaml_config(path: Optional[Path] = None) -> dict:
    """Load config.yaml from project root."""
    if path is None:
        path = Path(__file__).parent.parent / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
