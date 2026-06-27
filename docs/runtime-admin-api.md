# Runtime Admin API

This note documents the image-pool runtime API for inspecting models, loading
and unloading models, running image generation/edit requests, and starting LoRA
training jobs.

The goal is similar to the llm-pool runtime admin API: routine model management
should not require editing `local.json` and restarting the service.

It is intentionally a v1 design:

- live runtime control only
- no automatic writes back to `settings.json` or `local.json`
- no arbitrary model definitions via API
- no runtime load overrides in the load request body
- no background job system for model loads
- no force unload or graceful drain state yet

Current reality note:

- this admin API is implemented and is the live control plane used by the
  workbench
- model definitions still come from merged `settings.json + local.json`
- load/unload only changes in-process runtime state
- `recommended_steps` and `recommended_guidance` are the only generation
  defaults currently surfaced by model definitions
- richer per-model parameter schemas are proposed below, but are not implemented
  yet
- Flux and Z-Image LoRA training endpoints exist and share one in-process
  training slot

## Contents

- [Purpose](#purpose)
- [Core Concepts](#core-concepts)
- [State Semantics](#state-semantics)
- [Model Definition Fields](#model-definition-fields)
- [Endpoints](#endpoints)
  - [`GET /v1/models`](#get-v1models)
  - [`GET /v1/admin/models`](#get-v1adminmodels)
  - [`GET /v1/admin/gpu-memory`](#get-v1admingpu-memory)
  - [`POST /v1/admin/models/{model_name}/load`](#post-v1adminmodelsmodel_nameload)
  - [`POST /v1/admin/models/{model_name}/unload`](#post-v1adminmodelsmodel_nameunload)
  - [`POST /v1/images/generations`](#post-v1imagesgenerations)
  - [`POST /v1/images/edits`](#post-v1imagesedits)
  - [`GET /v1/training/flux-lora`](#get-v1trainingflux-lora)
  - [`POST /v1/training/flux-lora`](#post-v1trainingflux-lora)
  - [`POST /v1/training/flux-lora/stop`](#post-v1trainingflux-lorastop)
  - [`GET /v1/training/z-image-lora`](#get-v1trainingz-image-lora)
  - [`POST /v1/training/z-image-lora`](#post-v1trainingz-image-lora)
  - [`POST /v1/training/z-image-lora/stop`](#post-v1trainingz-image-lorastop)
- [Image Request Parameters](#image-request-parameters)
- [Proposed Parameter Schema](#proposed-parameter-schema)
- [Unload And In-Flight Requests](#unload-and-in-flight-requests)
- [Errors](#errors)

## Purpose

The current service merges `settings.json` and `local.json` into one effective
config, then loads enabled models at startup.

The admin API adds a separate live control plane on top of that merged config:

- the merged config tells us which models are known to the service
- the live runtime state tells us which of those models are currently loaded
- generation and edit endpoints accept only loaded models

That distinction must stay explicit in both the API and the UI.

## Core Concepts

### Configured Model Definition

A configured model definition comes from the merged `settings.json + local.json`
payload.

This is static process input. It includes fields such as:

- `backend`
- `enabled`
- `target_inflight`
- `model_path`
- `base_model_path`
- `transformer_config_path`
- `modalities`
- `output_modalities`
- `tasks`
- `max_images`
- `max_output_images`
- `vram_estimate_mib`
- `recommended_steps`
- `recommended_guidance`

This definition is not modified by the admin API in v1.

### Runtime State

Each configured model also has live runtime state inside the process.

Current states are represented by booleans:

- `loaded`
- `loading`

The GPU memory view maps those booleans to:

- `unloaded`
- `loading`
- `loaded`

There is no separate `unloading` state yet.

### Capabilities

Capabilities are derived from the model definition.

Current shape:

```json
{
  "input_modalities": ["text", "image"],
  "output_modalities": ["image"],
  "tasks": ["image_generation", "image_edit"],
  "max_images": 1,
  "max_output_images": 1
}
```

`tasks` is the field the UI should use to decide whether a model can be shown
for text-to-image, image-to-image/edit, or both.

## State Semantics

### `unloaded`

- the model exists in merged config
- no runtime is currently loaded
- image requests for this model are rejected
- the model may be loaded through the admin API

### `loading`

- a runtime load has started but is not complete yet
- image requests for this model are rejected
- duplicate load requests currently race with the in-progress load rather than
  joining a background job

### `loaded`

- a runtime exists and may serve image requests
- the model may be unloaded through the admin API

### Failed Load

Failed load attempts store `last_error` on the model state. The model remains
unloaded and may be loaded again through the admin API.

## Model Definition Fields

Example model definition:

```json
{
  "backend": "diffusers_z_image",
  "enabled": false,
  "target_inflight": 1,
  "model_path": "/home/gunnar/models/Z-Image-Turbo",
  "modalities": ["text", "image"],
  "output_modalities": ["image"],
  "tasks": ["image_generation", "image_edit"],
  "max_images": 1,
  "max_output_images": 1,
  "vram_estimate_mib": 16000,
  "recommended_steps": 9,
  "recommended_guidance": 0.0
}
```

Known backend ids:

- `stub`
- `diffusers_flux2_klein`
- `diffusers_firered_gguf`
- `diffusers_sdxl`
- `diffusers_z_image`

`vram_estimate_mib` is a configured estimate unless the runtime has observed a
load delta for the model during this process lifetime. See
`GET /v1/admin/gpu-memory`.

## Endpoints

### `GET /v1/models`

Returns loaded models only. This is the public model list for image requests.

Response shape:

```json
{
  "object": "list",
  "data": [
    {
      "id": "z-image-turbo",
      "object": "model",
      "owned_by": "image-pool",
      "backend": "diffusers_z_image",
      "capabilities": {
        "input_modalities": ["text", "image"],
        "output_modalities": ["image"],
        "tasks": ["image_generation", "image_edit"],
        "max_images": 1,
        "max_output_images": 1
      },
      "recommended_steps": 9,
      "recommended_guidance": 0.0
    }
  ]
}
```

### `GET /v1/admin/models`

Returns all known models from merged config together with their live runtime
state. This endpoint is the main UI source of truth for model management.

Response shape:

```json
{
  "object": "list",
  "data": [
    {
      "id": "z-image-turbo",
      "backend": "diffusers_z_image",
      "enabled": false,
      "loaded": true,
      "loading": false,
      "loaded_at": 1782500000.0,
      "last_error": null,
      "scheduler": {
        "target_inflight": 1,
        "inflight": 0,
        "queued": 0
      },
      "capabilities": {
        "input_modalities": ["text", "image"],
        "output_modalities": ["image"],
        "tasks": ["image_generation", "image_edit"],
        "max_images": 1,
        "max_output_images": 1
      },
      "model_path": "/home/gunnar/models/Z-Image-Turbo",
      "base_model_path": null,
      "vram_estimate_mib": 20600,
      "vram_estimate_source": "observed_load_delta",
      "recommended_steps": 9,
      "recommended_guidance": 0.0
    }
  ]
}
```

`vram_estimate_source` can currently be:

- `observed_load_delta`
- `configured`
- `model_artifact_size`
- `unavailable`

### `GET /v1/admin/gpu-memory`

Returns GPU memory data plus a model-oriented projection for the workbench model
table.

Response shape:

```json
{
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA ...",
      "memory_total_mib": 98304,
      "memory_used_mib": 68200,
      "memory_free_mib": 30104
    }
  ],
  "models": [
    {
      "name": "z-image-turbo",
      "runtime_state": "loaded",
      "is_loaded": true,
      "configured_target_inflight": 1,
      "effective_target_inflight": 1,
      "vram_estimate_mib": 20600,
      "vram_estimate_source": "observed_load_delta"
    }
  ],
  "error": null
}
```

If GPU probing fails, `gpus` may be empty and `error` contains the probe error.

### `POST /v1/admin/models/{model_name}/load`

Loads a configured model into the current process.

Request body: none.

Returns the same state object used by `GET /v1/admin/models`.

Notes:

- this does not change `enabled` in config
- this does not persist anything to `settings.json` or `local.json`
- if the model is already loaded, the current state is returned
- load failures are returned as `500 Internal Server Error` by FastAPI unless
  caught by a more specific handler

### `POST /v1/admin/models/{model_name}/unload`

Unloads a configured model from the current process.

Request body: none.

Returns the same state object used by `GET /v1/admin/models`.

Notes:

- this does not change `enabled` in config
- this does not persist anything to `settings.json` or `local.json`
- if the model is already unloaded, the current unloaded state is returned

### `POST /v1/images/generations`

Runs text-to-image generation.

Request shape:

```json
{
  "model": "z-image-turbo",
  "prompt": "A product photo of black running shoes on a concrete floor",
  "n": 1,
  "size": "1024x1024",
  "quality": "auto",
  "response_format": "b64_json",
  "seed": 1234,
  "allow_remote": false,
  "metadata": {
    "steps": 9,
    "guidance": 0.0
  }
}
```

Response shape:

```json
{
  "id": "img-...",
  "object": "image.generation",
  "created": 1782500000,
  "model": "z-image-turbo",
  "data": [
    {
      "b64_json": "...",
      "mime_type": "image/png",
      "revised_prompt": null
    }
  ],
  "metrics": {
    "steps": 9,
    "guidance": 0.0,
    "engine_queue_wait_ms": 0.2,
    "engine_total_wall_ms": 12345.0,
    "pool_total_wall_ms": 12345.5
  }
}
```

### `POST /v1/images/edits`

Runs image-to-image or image edit generation. The selected model must advertise
`image_edit` in `capabilities.tasks`.

Request shape:

```json
{
  "model": "sdxl-base-1.0",
  "prompt": "The same dog wearing sunglasses",
  "images": [
    {
      "name": "dog.png",
      "data_url": "data:image/png;base64,..."
    }
  ],
  "n": 1,
  "size": "1024x1024",
  "quality": "auto",
  "response_format": "b64_json",
  "seed": 1234,
  "allow_remote": false,
  "metadata": {
    "steps": 30,
    "guidance": 5.0,
    "strength": 0.35
  }
}
```

Response shape is the same as `POST /v1/images/generations`, with
`object: "image.edit"`.

### `GET /v1/training/flux-lora`

Returns Flux LoRA training backend availability and current run state.

Response shape:

```json
{
  "backend": {
    "id": "diffusers_flux2_lora",
    "label": "Flux",
    "available": true,
    "message": ""
  },
  "run": {
    "status": "idle",
    "run_id": "",
    "pid": null,
    "returncode": null,
    "started_at": "",
    "completed_at": "",
    "output_path": "",
    "log_tail": "",
    "message": "",
    "backend_id": "",
    "progress": {
      "step": 0,
      "steps": 0,
      "loss": null,
      "learning_rate": null
    }
  }
}
```

### `POST /v1/training/flux-lora`

Starts a Flux LoRA training run.

Request shape:

```json
{
  "model": "flux2-klein-base-4b",
  "dataset_path": "/home/gunnar/projects/llm-workbench/data/image_pool/training/datasets/example",
  "output_path": "/home/gunnar/projects/llm-workbench/data/image_pool/training/flux2-klein/runs",
  "trigger_word": "GFX_IMPR5N",
  "steps": 3000,
  "learning_rate": 0.000095,
  "rank": 128,
  "alpha": 64,
  "batch_size": 1,
  "checkpoint_interval": 500,
  "resolution": [256, 512, 768, 1024, 1280, 1536],
  "metadata": {}
}
```

Returns the same shape as `GET /v1/training/flux-lora`.

### `POST /v1/training/flux-lora/stop`

Requests stop for the active training run. The worker finishes the current step
before stopping when possible.

Returns the same shape as `GET /v1/training/flux-lora`.

### `GET /v1/training/z-image-lora`

Returns Z-Image LoRA training backend availability and current run state.

Response shape is the same as `GET /v1/training/flux-lora`, with
`backend.id: "diffusers_z_image_lora"`.

### `POST /v1/training/z-image-lora`

Starts a Z-Image LoRA training run.

Request shape:

```json
{
  "model": "z-image-base",
  "dataset_path": "/home/gunnar/projects/llm-workbench/data/image_pool/training/datasets/example",
  "output_path": "/home/gunnar/projects/llm-workbench/data/image_pool/training/z-image/runs",
  "trigger_word": "GFX_IMPR5N",
  "steps": 500,
  "learning_rate": 0.0001,
  "rank": 4,
  "alpha": 4,
  "batch_size": 1,
  "checkpoint_interval": 500,
  "resolution": 1024,
  "metadata": {}
}
```

Returns the same shape as `GET /v1/training/z-image-lora`.

### `POST /v1/training/z-image-lora/stop`

Requests stop for the active training run. The worker finishes the current step
before stopping when possible.

Returns the same shape as `GET /v1/training/z-image-lora`.

## Image Request Parameters

The common request fields are currently:

- `model`: configured model id
- `prompt`: positive prompt
- `n`: output image count, currently 1 to 4 by schema and additionally limited
  by model `max_output_images`
- `size`: output size string such as `1024x1024`
- `quality`: `auto`, `low`, `medium`, or `high`
- `response_format`: currently only `b64_json`
- `seed`: optional integer seed
- `allow_remote`: reserved request flag
- `metadata`: backend-specific runtime parameters

Current metadata keys used by backends:

- `steps`: integer, 1 to 80
- `guidance`: number, 0.0 to 20.0
- `negative_prompt`: SDXL only
- `strength`: image edit/img2img strength, 0.0 to 1.0
- `lora_id`: optional UI/runtime identifier for metrics
- `lora_path`: path to `.safetensors` LoRA weights
- `lora_scale`: LoRA adapter weight, 0.0 to 2.0

Current backend defaults:

| Backend | Steps | Guidance | Strength | Notes |
| --- | ---: | ---: | ---: | --- |
| `diffusers_flux2_klein` | 4 | 1.0 | n/a | Supports LoRA metadata |
| `diffusers_sdxl` | `recommended_steps` or 30 | `recommended_guidance` or 5.0 | 0.35 | Supports `negative_prompt` |
| `diffusers_z_image` | `recommended_steps` or 9 | `recommended_guidance` or 0.0 | 0.35 | Supports LoRA metadata |
| `diffusers_firered_gguf` | 40 | 4.0 | n/a | Image edit backend |

## Proposed Parameter Schema

The workbench needs enough model-specific parameter metadata to build the image
generation details section for the selected model and to implement `Reset
defaults` without hardcoding per-model rules in the UI.

Capabilities should answer what the model can do. Parameter schemas should
answer which controls the UI should show for that model.

Proposed model definition extension:

```json
{
  "generation_parameters": {
    "size": {
      "kind": "enum",
      "default": "1024x1024",
      "allowed_values": ["768x768", "1024x1024", "1024x1536", "1536x1024"]
    },
    "steps": {
      "kind": "integer",
      "default": 9,
      "minimum": 1,
      "maximum": 80,
      "step": 1
    },
    "guidance": {
      "kind": "number",
      "default": 0.0,
      "minimum": 0.0,
      "maximum": 20.0,
      "step": 0.1
    },
    "seed": {
      "kind": "integer_or_null",
      "default": null,
      "minimum": 0,
      "step": 1
    }
  },
  "edit_parameters": {
    "strength": {
      "kind": "number",
      "default": 0.35,
      "minimum": 0.0,
      "maximum": 1.0,
      "step": 0.05
    }
  }
}
```

Suggested API behavior once implemented:

- `GET /v1/models` includes parameter schemas for loaded models
- `GET /v1/admin/models` includes the same parameter schemas plus runtime state
- the workbench renders controls from these schemas
- `Reset defaults` applies the selected model's schema defaults
- existing `recommended_steps` and `recommended_guidance` can be removed after
  the workbench no longer depends on them

Suggested UI mapping:

- `integer` -> stepper or numeric input
- `number` -> slider plus numeric input where precision matters
- `enum` -> select, menu, or segmented control
- `integer_or_null` -> numeric input with an unset/locked state

## Unload And In-Flight Requests

Current unload behavior is immediate:

- the model executor is unregistered
- queued requests are failed with `model_not_loaded`
- worker tasks are cancelled
- the runtime is closed
- Torch CUDA cache is released

There is no graceful drain mode yet. If the UI needs safer unload behavior
during active generation, image-pool should grow an explicit `unloading` state
and reject new requests while allowing in-flight requests to finish.

## Errors

Current explicit error handlers:

| Condition | HTTP status | Error type |
| --- | ---: | --- |
| Unknown model | 404 | `unknown_model` |
| Model not loaded | 409 | `model_not_loaded` |
| Unsupported backend | 400 | `unsupported_backend` |
| Bad request / validation-like runtime error | 400 | `bad_request` |

Training preflight errors are returned as `HTTPException` details. Common
training error ids:

- `training_dataset_not_ready`
- `training_model_not_ready`
- `training_backend_unavailable`
- `training_run_active`
