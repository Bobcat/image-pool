from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.config import ModelSettings, Settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPORTED_LORAS_ROOT = PROJECT_ROOT / "data" / "loras" / "imported"
IMPORTED_LORA_WEIGHT_NAME = "adapter.safetensors"


class LoraInspectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(min_length=1)


class LoraImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(min_length=1)
    name: str = Field(min_length=1)
    family: str = Field(min_length=1)
    compatible_models: list[str] = Field(min_length=1)
    trained_on_model_id: str = ""
    trigger_words: list[str] = Field(default_factory=list)
    default_strength: float | None = Field(default=None, ge=0.0, le=2.0)
    description: str = ""
    source_url: str = ""


def list_loras() -> dict[str, object]:
    return {"object": "list", "data": _imported_lora_payloads()}


def inspect_lora(settings: Settings, request: LoraInspectRequest) -> dict[str, object]:
    source_path = _validated_lora_path(request.source_path)
    metadata, keys = _read_safetensors_header(source_path)
    detected_modules = _detected_modules(keys)
    family_guess, confidence = _family_guess(metadata, keys)
    model_options = _model_options(settings)
    compatible_suggestions = [
        item["id"]
        for item in model_options
        if item["supports_lora"] and item["family"] == family_guess
    ]
    warnings = []
    if not family_guess:
        warnings.append("Could not infer the LoRA family from safetensors metadata or keys.")
    if family_guess and not compatible_suggestions:
        warnings.append("No configured LoRA-capable model matches the inferred family.")

    return {
        "filename": source_path.name,
        "source_path": str(source_path),
        "size_bytes": source_path.stat().st_size,
        "metadata": metadata,
        "key_count": len(keys),
        "detected_modules": detected_modules,
        "family_guess": family_guess,
        "confidence": confidence,
        "trigger_words": _metadata_trigger_words(metadata),
        "default_strength_suggestion": _default_strength_suggestion(settings, compatible_suggestions),
        "compatible_model_suggestions": compatible_suggestions,
        "model_options": model_options,
        "warnings": warnings,
    }


def import_lora(settings: Settings, request: LoraImportRequest) -> dict[str, object]:
    source_path = _validated_lora_path(request.source_path)
    _read_safetensors_header(source_path)
    _validate_import_request(settings, request)

    slug = _unique_slug(_slugify(request.name, fallback="lora"))
    lora_dir = IMPORTED_LORAS_ROOT / slug
    lora_dir.mkdir(parents=True)
    weight_path = lora_dir / IMPORTED_LORA_WEIGHT_NAME
    shutil.copy2(source_path, weight_path)
    (lora_dir / "metadata.json").write_text(
        json.dumps(
            {
                "name": request.name.strip(),
                "family": request.family.strip(),
                "trained_on_model_id": request.trained_on_model_id.strip(),
                "compatible_models": _clean_string_list(request.compatible_models),
                "trigger_words": _clean_string_list(request.trigger_words),
                "default_strength": request.default_strength,
                "description": request.description.strip(),
                "source_url": request.source_url.strip(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"lora": _imported_lora_payload(lora_dir)}


def _validate_import_request(settings: Settings, request: LoraImportRequest) -> None:
    family = request.family.strip()
    if not family:
        raise ValueError("LoRA family is required.")
    compatible_models = _clean_string_list(request.compatible_models)
    if not compatible_models:
        raise ValueError("At least one compatible model is required.")

    options_by_id = {item["id"]: item for item in _model_options(settings)}
    for model_id in compatible_models:
        option = options_by_id.get(model_id)
        if option is None:
            raise ValueError(f"Unknown compatible model: {model_id}")
        if not option["supports_lora"]:
            raise ValueError(f"Model does not support LoRA loading: {model_id}")
        if option["family"] != family:
            raise ValueError(f"Model {model_id} is not in LoRA family {family}.")


def _validated_lora_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"LoRA file does not exist: {path}")
    if path.suffix.lower() != ".safetensors":
        raise ValueError("LoRA file must be a .safetensors file.")
    return path


def _read_safetensors_header(path: Path) -> tuple[dict[str, str], list[str]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ValueError("safetensors is required to inspect LoRA files.") from exc

    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            keys = list(handle.keys())
    except Exception as exc:
        raise ValueError(f"Could not read LoRA safetensors file: {exc}") from exc
    return {str(key): str(value) for key, value in metadata.items()}, keys


def _imported_lora_payloads() -> list[dict[str, object]]:
    if not IMPORTED_LORAS_ROOT.is_dir():
        return []
    payloads = []
    for lora_dir in sorted(item for item in IMPORTED_LORAS_ROOT.iterdir() if item.is_dir()):
        payload = _imported_lora_payload(lora_dir)
        if payload:
            payloads.append(payload)
    return payloads


def _imported_lora_payload(lora_dir: Path) -> dict[str, object]:
    weight_path = lora_dir / IMPORTED_LORA_WEIGHT_NAME
    if not weight_path.is_file():
        candidates = sorted(lora_dir.glob("*.safetensors"))
        if not candidates:
            return {}
        weight_path = candidates[0]
    metadata = _read_json_object(lora_dir / "metadata.json")
    slug = lora_dir.name
    trigger_words = _clean_string_list(metadata.get("trigger_words"))
    compatible_models = _clean_string_list(metadata.get("compatible_models"))
    default_strength = _float_or_none(metadata.get("default_strength"))
    return {
        "id": f"imported/{slug}",
        "name": str(metadata.get("name") or _name_from_slug(slug)).strip(),
        "family": str(metadata.get("family") or "").strip(),
        "source_type": "imported",
        "artifact_type": "imported",
        "run_id": "",
        "dataset": "",
        "trained_on_model_id": str(metadata.get("trained_on_model_id") or "").strip(),
        "model": str(metadata.get("trained_on_model_id") or "").strip(),
        "compatible_models": compatible_models,
        "trigger_words": trigger_words,
        "trigger_word": trigger_words[0] if trigger_words else "",
        "default_strength": default_strength,
        "description": str(metadata.get("description") or "").strip(),
        "source_url": str(metadata.get("source_url") or "").strip(),
        "path": str(weight_path.resolve()),
        "display_path": _display_path(weight_path),
        "size_bytes": weight_path.stat().st_size,
        "kind": "imported",
        "checkpoint_id": "",
        "checkpoint_step": None,
    }


def _model_options(settings: Settings) -> list[dict[str, object]]:
    options = []
    for model_id, model_settings in sorted(settings.engine.models.items()):
        family = _model_family(model_id, model_settings)
        options.append(
            {
                "id": model_id,
                "family": family,
                "backend": model_settings.backend,
                "supports_lora": _model_supports_lora(model_settings),
            }
        )
    return options


def _model_family(model_id: str, settings: ModelSettings) -> str:
    if settings.backend == "diffusers_flux2_klein" or model_id.startswith("flux2-klein"):
        return "flux2-klein"
    if settings.backend == "diffusers_z_image" or model_id.startswith("z-image"):
        return "z-image"
    if settings.backend == "diffusers_sdxl" or model_id.startswith("sdxl"):
        return "sdxl"
    return ""


def _model_supports_lora(settings: ModelSettings) -> bool:
    if settings.backend not in {"diffusers_flux2_klein", "diffusers_z_image", "diffusers_sdxl"}:
        return False
    return "lora_scale" in settings.generation_parameters or "lora_scale" in settings.edit_parameters


def _family_guess(metadata: dict[str, str], keys: list[str]) -> tuple[str, float]:
    metadata_text = " ".join(f"{key} {value}" for key, value in metadata.items()).lower()
    key_text = "\n".join(keys[:400]).lower()
    combined = f"{metadata_text}\n{key_text}"
    if "sdxl" in combined or "lora_unet" in combined or ".unet." in combined or "text_encoder_2" in combined:
        return "sdxl", 0.85
    if "single_transformer_blocks" in combined or "to_qkv_mlp_proj" in combined:
        return "flux2-klein", 0.75
    if "z-image" in combined or "transformer_blocks" in combined:
        return "z-image", 0.65
    return "", 0.0


def _detected_modules(keys: list[str]) -> list[str]:
    modules = set()
    for key in keys:
        lowered = key.lower()
        if "unet" in lowered:
            modules.add("unet")
        if "text_encoder_2" in lowered:
            modules.add("text_encoder_2")
        elif "text_encoder" in lowered:
            modules.add("text_encoder")
        if "transformer" in lowered:
            modules.add("transformer")
    return sorted(modules)


def _metadata_trigger_words(metadata: dict[str, str]) -> list[str]:
    for key in ("trigger_words", "trigger_word", "ss_trigger_words", "trained_words", "trainedWords"):
        value = metadata.get(key)
        words = _split_words(value)
        if words:
            return words
    return []


def _default_strength_suggestion(settings: Settings, model_ids: list[str]) -> float | None:
    for model_id in model_ids:
        model_settings = settings.engine.models.get(model_id)
        if model_settings is None:
            continue
        for schema in (model_settings.generation_parameters, model_settings.edit_parameters):
            lora_scale = schema.get("lora_scale") if isinstance(schema, dict) else None
            if isinstance(lora_scale, dict):
                parsed = _float_or_none(lora_scale.get("default"))
                if parsed is not None:
                    return parsed
    return None


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _clean_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return _split_words(value)
    return []


def _split_words(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip() for part in re.split(r"[,;\n]+", text) if part.strip()]


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _unique_slug(base_slug: str) -> str:
    IMPORTED_LORAS_ROOT.mkdir(parents=True, exist_ok=True)
    slug = base_slug
    index = 2
    while (IMPORTED_LORAS_ROOT / slug).exists():
        slug = f"{base_slug}-{index}"
        index += 1
    return slug


def _slugify(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return slug[:80] or fallback


def _name_from_slug(slug: str) -> str:
    return " ".join(part for part in slug.replace("_", "-").split("-") if part).title() or "Imported LoRA"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
