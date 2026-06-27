# image-pool

`image-pool` is a small FastAPI service for local image generation and image
editing backends. It exposes an OpenAI-style image API, keeps model lifecycle
state in one process, and lets a UI or another service load, unload, inspect,
call local image models, and start local LoRA training runs.

The service is intentionally narrow: it owns local image model routing and
runtime scheduling. It does not own UI workflows, persistent artifact storage,
or long-term job persistence.

## Index

- [What It Does](#what-it-does)
- [Repository Role](#repository-role)
- [Related Repositories](#related-repositories)
- [Code Map](#code-map)
- [API Surface](#api-surface)
- [Runtime Model](#runtime-model)
- [Configuration](#configuration)
- [Model Directories](#model-directories)
- [Backends](#backends)
- [Development](#development)
- [Tests](#tests)
- [Deployment Notes](#deployment-notes)
- [License](#license)

## What It Does

- Provides text-to-image requests through `POST /v1/images/generations`.
- Provides image-edit requests through `POST /v1/images/edits`.
- Reports currently loaded public models through `GET /v1/models`.
- Reports all configured models and runtime state through `GET /v1/admin/models`.
- Loads and unloads configured models with admin endpoints.
- Reports coarse GPU memory information from `nvidia-smi`.
- Runs one in-process scheduler per loaded model with a configurable
  `target_inflight`.
- Starts and monitors local LoRA training runs for FLUX.2-klein and Z-Image
  models.

## Repository Role

This repo owns:

- The image-pool HTTP API.
- Model config loading and `config/local.json` overrides.
- In-process model lifecycle and scheduling.
- Local Diffusers-based image generation/editing runtimes.
- Local LoRA training workers for supported image backends.
- A stub runtime for API and lifecycle tests.

This repo deliberately does not own:

- Browser UI. The current UI lives in `llm-workbench`.
- Persistent image artifact storage.
- A queue that survives process restart.

## Related Repositories

- `llm-workbench`: browser UI and proxy endpoints for image-pool.
- `llm-pool`, `tts-pool`, `asr-pool`: sibling local pool services with similar
  lifecycle ideas, but different model domains.

## Code Map

```text
app/main.py
  FastAPI app, routes, lifespan, and error mapping.

app/config.py
  Pydantic settings models and config/local.json merge logic.

app/schemas.py
  Request and response schemas for image generation, image editing, and LoRA
  training.

app/engine/router.py
  Model registry, load/unload logic, public/admin payloads, GPU memory payloads.

app/engine/scheduler.py
  Per-model in-process queue and target_inflight worker control.

app/engine/stub.py
  Test backend that returns generated PNG payloads.

app/engine/diffusers_flux.py
  FLUX.2-klein Diffusers runtime for text-to-image and image edit.

app/engine/flux_fp8.py
  Helpers for loading FLUX.2-klein FP8 safetensor variants with a base pipeline.

app/engine/diffusers_sdxl.py
  SDXL Diffusers runtime for text-to-image and img2img-style editing.

app/engine/diffusers_z_image.py
  Z-Image Diffusers runtime for text-to-image, img2img, and LoRA adapter use.

app/engine/diffusers_firered_gguf.py
  FireRed/Qwen image-edit runtime using a GGUF transformer and Diffusers.

app/training.py
  In-process FLUX.2-klein and Z-Image LoRA training workers and status state.

config/settings.json
  Base model and service configuration.

docs/runtime-admin-api.md
  Detailed runtime admin API notes, including current payloads and proposed
  parameter schema extensions.

tests/
  Lightweight API and configuration tests.
```

## API Surface

For the full runtime/admin contract, see
[`docs/runtime-admin-api.md`](docs/runtime-admin-api.md).

### Health

```http
GET /healthz
```

Returns:

```json
{"status": "ok"}
```

### Public Models

```http
GET /v1/models
```

Returns only loaded models. Each model includes its backend and capabilities.

### Admin Models

```http
GET /v1/admin/models
```

Returns all configured models, including unloaded models, scheduler state,
configured model paths, capabilities, load errors, and VRAM estimates.

### GPU Memory

```http
GET /v1/admin/gpu-memory
```

Returns GPU memory data from `nvidia-smi` plus configured model estimates.

### Load A Model

```http
POST /v1/admin/models/{model_name}/load
```

Loads a configured model into the process and registers its scheduler.

### Unload A Model

```http
POST /v1/admin/models/{model_name}/unload
```

Unregisters the model scheduler and releases the runtime object. CUDA allocators
may keep reserved memory until process restart depending on the backend.

### Generate Images

```bash
curl -s http://127.0.0.1:8013/v1/images/generations \
  -H 'content-type: application/json' \
  -d '{
    "model": "flux2-klein-4b",
    "prompt": "paint the Eiffel Tower by night",
    "size": "512x512",
    "n": 1,
    "metadata": {
      "steps": 4,
      "guidance": 1.0
    }
  }'
```

The response contains base64 PNG data in `data[].b64_json`.

### Edit Images

```bash
curl -s http://127.0.0.1:8013/v1/images/edits \
  -H 'content-type: application/json' \
  -d '{
    "model": "flux2-klein-4b",
    "prompt": "remove all text from the package label",
    "size": "512x512",
    "n": 1,
    "metadata": {
      "steps": 4,
      "guidance": 1.0
    },
    "images": [
      {
        "name": "input.png",
        "data_url": "data:image/png;base64,..."
      }
    ]
  }'
```

`images[].data_url` must be an image data URL with a base64 payload.

### LoRA Training

```http
GET /v1/training/flux-lora
POST /v1/training/flux-lora
POST /v1/training/flux-lora/stop
```

```http
GET /v1/training/z-image-lora
POST /v1/training/z-image-lora
POST /v1/training/z-image-lora/stop
```

Training requests point at an existing dataset directory and output directory.
The dataset directory must contain image files with matching `.txt` captions.
The service keeps one in-process training state, so only one training run is
active at a time.

## Runtime Model

`image-pool` is a single-process service. At startup it reads settings, creates
model state entries, and loads models where `enabled` is `true`.

Each loaded model gets a `LoadedModelExecutor` in the scheduler. Requests for a
model enter that model's queue and are processed up to `target_inflight` at a
time. Current real image backends are configured with `target_inflight: 1`,
which avoids concurrent GPU work inside one model runtime.

Model load and unload are runtime actions. They do not rewrite config files.
The `enabled` field controls startup behavior only.

Training runs execute inside the service process in a worker thread and write
their outputs to the requested output directory. Training status is runtime
state; it is not a durable job queue and does not survive process restart.

## Configuration

Base settings live in:

```text
config/settings.json
```

Machine-local overrides can be placed in:

```text
config/local.json
```

`config/local.json` is ignored by git. It is the right place to change local
model paths, enable/disable models on one machine, or tune VRAM estimates.

Important model fields:

| Field | Meaning |
|---|---|
| `backend` | Runtime backend key, such as `stub`, `diffusers_flux2_klein`, or `diffusers_firered_gguf`. |
| `enabled` | Load this model automatically at service startup. |
| `target_inflight` | Maximum concurrent requests for the loaded model executor. |
| `model_path` | Local model directory or file used by the backend. |
| `base_model_path` | Optional local base pipeline directory for backends that need a separate base model. |
| `transformer_config_path` | Optional transformer config path for backends that load a separate transformer artifact. |
| `modalities` | Input modalities, for example `["text", "image"]`. |
| `output_modalities` | Output modalities, currently `["image"]`. |
| `tasks` | Supported tasks, such as `image_generation` and `image_edit`. |
| `max_images` | Maximum input images accepted by image-edit requests. |
| `max_output_images` | Maximum output images per request. |
| `vram_estimate_mib` | Configured VRAM estimate shown by admin/UI surfaces. |
| `recommended_steps` | Model-specific default step count for UI/runtime callers. |
| `recommended_guidance` | Model-specific default guidance value for UI/runtime callers. |

## Model Directories

Prefer readable local model directories and files over Hugging Face cache
`blobs/refs/snapshots` paths. A local Diffusers model directory should contain
files such as `model_index.json`, `transformer/`, `vae/`, `text_encoder/`, and
tokenizer or processor directories.

Example local layout:

```text
/path/to/models/
  FLUX.2-klein-4B/
    model_index.json
    transformer/
    vae/
    text_encoder/
    tokenizer/

  FireRed-Image-Edit-1.1/
    model_index.json
    transformer/config.json
    text_encoder/
    vae/
    processor/
    tokenizer/

  FireRed-Image-Edit-1.1-Q4_K_M.gguf

  stable-diffusion-xl-base-1.0/
    model_index.json
    unet/
    vae/
    text_encoder/
    text_encoder_2/
    tokenizer/
    tokenizer_2/

  Z-Image-Turbo/
    model_index.json
    transformer/
    vae/
    text_encoder/
    tokenizer/
```

Example download commands:

```bash
huggingface-cli download black-forest-labs/FLUX.2-klein-4B \
  --local-dir /path/to/models/FLUX.2-klein-4B
```

```bash
huggingface-cli download FireRedTeam/FireRed-Image-Edit-1.1 \
  --local-dir /path/to/models/FireRed-Image-Edit-1.1
```

```bash
huggingface-cli download vantagewithai/FireRed-Image-Edit-1.1-GGUF \
  FireRed-Image-Edit-1.1-Q4_K_M.gguf \
  --local-dir /path/to/models
```

Then point `model_path` and `base_model_path` at those local paths.

## Backends

### `stub`

The stub backend is enabled by default. It validates request shape and returns
small PNG payloads. It is used for API and scheduler tests and does not require
CUDA.

### `diffusers_flux2_klein`

The FLUX.2-klein backend uses `Flux2KleinPipeline` from Diffusers.

Capabilities:

- Text-to-image.
- Image edit.
- Up to 4 input images in config.
- One output image per request in config.

Defaults:

- `steps`: `4`
- `guidance`: `1.0`
- `torch_dtype`: `bfloat16`
- Device: CUDA

The current runtime loads the full pipeline onto GPU.

FP8 FLUX.2-klein safetensor variants can be configured with `model_path`
pointing at the safetensor file and `base_model_path` pointing at the matching
Diffusers base pipeline directory.

### `diffusers_sdxl`

The SDXL backend uses `StableDiffusionXLPipeline` for text-to-image and
`StableDiffusionXLImg2ImgPipeline` when an input image is provided.

Capabilities:

- Text-to-image.
- Img2img-style image edit.
- One input image in config.
- One output image per request in config.

Defaults:

- `steps`: `recommended_steps` or `30`
- `guidance`: `recommended_guidance` or `5.0`
- `strength`: `0.35` for image edit requests
- Device: CUDA

### `diffusers_z_image`

The Z-Image backend uses the configured Z-Image Diffusers pipeline for
text-to-image and image-to-image requests.

Capabilities:

- Text-to-image.
- Img2img-style image edit when the pipeline supports it.
- LoRA adapter loading through request metadata.
- One input image in config.
- One output image per request in config.

Defaults:

- `steps`: `recommended_steps` or `9`
- `guidance`: `recommended_guidance` or `0.0`
- `strength`: `0.35` for image edit requests
- Device: CUDA

### `diffusers_firered_gguf`

The FireRed backend uses a Qwen-image edit pipeline with a GGUF transformer
file and a separate local base pipeline directory.

Capabilities:

- Image edit only.
- One input image in config.
- One output image per request in config.

Defaults:

- `steps`: `40`
- `guidance`: `4.0`, mapped to `true_cfg_scale`
- `negative_prompt`: a single space
- `torch_dtype`: `bfloat16`
- Device: CUDA

The runtime enables VAE tiling and slicing. The current version is technically
working, but should be treated as experimental: it has shown weaker edit
quality than the FLUX.2-klein backend for product/label editing tests.

## Development

Create an environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
```

For real Diffusers backends, install the optional runtime dependencies:

```bash
pip install -e '.[flux]'
```

The optional extra is currently named `flux`, but it contains the shared
Diffusers, Torch, PEFT, GGUF, and image runtime dependencies used by the real
backends.

Run the service:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8013
```

Or:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8013
```

## Tests

Run the lightweight test suite:

```bash
python -m pytest
```

Compile-check application and tests:

```bash
python -m compileall -q app tests
```

The unit tests do not load real Diffusers models. Real backend verification is
manual and should use the admin load endpoint plus a small generation or edit
request.

## Deployment Notes

There are no deployment scripts in this repo yet. Run it as a normal ASGI
service behind the local tooling that owns process supervision.

Operational notes:

- Keep `target_inflight` at `1` for large local image models unless the backend
  has been measured under concurrency.
- Prefer local model directories over Hugging Face cache blob paths.
- Restarting the process is the most reliable way to release all CUDA allocator
  state after heavy model experiments.

## License

No license file is currently present in this repository.
