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

    # SambaNova (PRIMARY — free, vision via Maverick, best performer)
    sambanova_api_key: str = ""
    sambanova_model: str = "Llama-4-Maverick-17B-128E-Instruct"

    # Groq (SECONDARY — free, fast, 109B MoE vision)
    groq_api_key: str = ""
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # OpenRouter (TERTIARY — fallback, vision-capable)
    openrouter_api_key: str = ""
    openrouter_model: str = "nvidia/nemotron-nano-12b-v2-vl:free"
    openrouter_paid_model: str = ""  # optional paid model

    # Google Gemini (QUATERNARY — free tier has low daily quota)
    google_api_key: str = ""
    google_model: str = "gemini-2.5-flash-lite"
    google_fallback_model: str = "gemini-2.5-flash"

    # Cerebras (disabled — llama3.1-8b too small for browser-use)
    cerebras_api_key: str = ""
    cerebras_model: str = "llama3.1-8b"

    openrouter_backup_models: str = (
        "google/gemma-3-12b-it:free,"
        "google/gemma-3-27b-it:free,"
        "deepseek/deepseek-r1-0528:free,"
        "z-ai/glm-4.5-air:free,"
        "arcee-ai/trinity-mini:free,"
        "stepfun/step-3.5-flash:free"
    )

    # Evaluator (single call, OpenRouter is fine)
    openrouter_eval_model: str = "openrouter/aurora-alpha"

    # Environment
    log_level: str = "INFO"

    @property
    def openrouter_backup_models_list(self) -> list[str]:
        """Return OpenRouter backup models as a list."""
        return [m.strip() for m in self.openrouter_backup_models.split(",") if m.strip()]

    def build_model_chain(self) -> list[str]:
        """Build ordered list of model IDs based on available API keys.

        Priority: SambaNova > Groq > OpenRouter > Google Gemini > OpenRouter backups.
        """
        chain: list[str] = []
        if self.sambanova_api_key:
            chain.append(self.sambanova_model)
        if self.groq_api_key:
            chain.append(self.groq_model)
        if self.openrouter_api_key and self.openrouter_model:
            chain.append(self.openrouter_model)
        if self.openrouter_api_key and self.openrouter_paid_model:
            chain.append(self.openrouter_paid_model)
        if self.google_api_key:
            chain.append(self.google_model)
            chain.append(self.google_fallback_model)
        if self.cerebras_api_key:
            chain.append(self.cerebras_model)
        if self.openrouter_api_key:
            chain.extend(self.openrouter_backup_models_list)
        return chain


def load_yaml_config(path: Optional[Path] = None) -> dict:
    """Load config.yaml from project root."""
    if path is None:
        path = Path(__file__).parent.parent / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
