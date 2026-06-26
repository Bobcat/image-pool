from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    data_url: str


class ImageGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: str = Field(min_length=1)
    n: int = Field(default=1, ge=1, le=4)
    size: str = "512x512"
    quality: Literal["auto", "low", "medium", "high"] = "auto"
    response_format: Literal["b64_json"] = "b64_json"
    seed: int | None = None
    allow_remote: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageEditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: str = Field(min_length=1)
    images: list[ImageInput] = Field(default_factory=list)
    n: int = Field(default=1, ge=1, le=4)
    size: str = "512x512"
    quality: Literal["auto", "low", "medium", "high"] = "auto"
    response_format: Literal["b64_json"] = "b64_json"
    seed: int | None = None
    allow_remote: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    b64_json: str
    mime_type: str = "image/png"
    revised_prompt: str | None = None


class ImageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["image.generation", "image.edit"]
    created: int
    model: str
    data: list[ImageData]
    metrics: dict[str, Any] = Field(default_factory=dict)


class FluxLoraTrainingStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    dataset_path: str = Field(min_length=1)
    output_path: str = Field(min_length=1)
    trigger_word: str = Field(default="GFX_IMPR5N", min_length=1)
    steps: int = Field(default=3000, ge=1)
    learning_rate: float = Field(default=0.000095, gt=0)
    rank: int = Field(default=128, ge=1)
    alpha: int = Field(default=64, ge=1)
    batch_size: int = Field(default=1, ge=1)
    resolution: list[int] = Field(
        default_factory=lambda: [256, 512, 768, 1024, 1280, 1536],
        min_length=1,
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
