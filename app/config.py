from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"


class ServiceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = 8013
    log_level: str = "info"


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "stub"
    enabled: bool = True
    target_inflight: int = Field(default=1, ge=1)
    model_path: str | None = None
    base_model_path: str | None = None
    modalities: tuple[str, ...] = ("text",)
    output_modalities: tuple[str, ...] = ("image",)
    tasks: tuple[str, ...] = ("image_generation",)
    max_images: int = Field(default=4, ge=0)
    max_output_images: int = Field(default=1, ge=1, le=8)
    vram_estimate_mib: int | None = Field(default=None, ge=0)
    recommended_steps: int | None = Field(default=None, ge=1)
    recommended_guidance: float | None = Field(default=None, ge=0)


class EngineSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "stub"
    models: dict[str, ModelSettings] = Field(default_factory=dict)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service: ServiceSettings = Field(default_factory=ServiceSettings)
    engine: EngineSettings = Field(default_factory=EngineSettings)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"settings file must contain a JSON object: {path}")
    return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(settings_path: str | Path | None = None) -> Settings:
    path = Path(settings_path) if settings_path is not None else DEFAULT_SETTINGS_PATH
    payload = _read_json(path)
    local_payload = _read_json(path.with_name("local.json"))
    return Settings.model_validate(_deep_merge(payload, local_payload))
