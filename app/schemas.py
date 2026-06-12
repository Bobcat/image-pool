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
