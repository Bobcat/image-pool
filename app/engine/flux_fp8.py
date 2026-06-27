from __future__ import annotations

from pathlib import Path
from typing import Any


_TOP_LEVEL_LINEAR_MAP = {
    "img_in": "x_embedder",
    "txt_in": "context_embedder",
    "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
    "single_stream_modulation.lin": "single_stream_modulation.linear",
    "final_layer.linear": "proj_out",
}

_DOUBLE_BLOCK_LINEAR_MAP = {
    "img_attn.proj": "attn.to_out.0",
    "img_mlp.0": "ff.linear_in",
    "img_mlp.2": "ff.linear_out",
    "txt_attn.proj": "attn.to_add_out",
    "txt_mlp.0": "ff_context.linear_in",
    "txt_mlp.2": "ff_context.linear_out",
}

_SINGLE_BLOCK_LINEAR_MAP = {
    "linear1": "attn.to_qkv_mlp_proj",
    "linear2": "attn.to_out",
}


def load_flux2_fp8_transformer(transformer_path: str | Path, config_path: str | Path, *, dtype):
    import torch
    from diffusers import Flux2Transformer2DModel
    from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
    from safetensors.torch import load_file

    source_path = Path(transformer_path).expanduser()
    config_root = Path(config_path).expanduser()
    source_state_dict = load_file(str(source_path), device="cpu")
    filtered_state_dict = {
        key: value
        for key, value in source_state_dict.items()
        if not key.endswith(".input_scale") and not key.endswith(".weight_scale")
    }
    state_dict = convert_flux2_transformer_checkpoint_to_diffusers(filtered_state_dict)
    _copy_scaled_fp8_params(source_state_dict, state_dict)

    config = Flux2Transformer2DModel.load_config(str(config_root), subfolder="transformer", local_files_only=True)
    if isinstance(config, tuple):
        config = config[0]
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        transformer = Flux2Transformer2DModel.from_config(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    fp8_target_names = _install_scaled_fp8_linears(transformer, state_dict)
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Could not load FP8 Flux transformer cleanly: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    _cast_non_fp8_parameters(transformer, dtype)
    transformer._keep_in_fp32_modules = tuple(fp8_target_names)
    return transformer


def is_flux2_fp8_transformer_path(path: str | Path | None) -> bool:
    if not path:
        return False
    return Path(path).expanduser().is_file() and Path(path).suffix.lower() == ".safetensors"


def _copy_scaled_fp8_params(source_state_dict: dict[str, Any], target_state_dict: dict[str, Any]) -> None:
    for key, value in source_state_dict.items():
        if key.endswith(".weight_scale"):
            base_key = key[: -len(".weight_scale")]
            suffix = "weight_scale"
        elif key.endswith(".input_scale"):
            base_key = key[: -len(".input_scale")]
            suffix = "input_scale"
        else:
            continue
        for target in _scaled_fp8_target_names(base_key):
            target_state_dict[f"{target}.{suffix}"] = value


def _scaled_fp8_target_names(source_name: str) -> list[str]:
    if source_name in _TOP_LEVEL_LINEAR_MAP:
        return [_TOP_LEVEL_LINEAR_MAP[source_name]]

    parts = source_name.split(".")
    if len(parts) < 3:
        return []

    if parts[0] == "double_blocks":
        block_index = parts[1]
        modality = parts[2]
        within_block = ".".join(parts[2:])
        if within_block.endswith(".qkv"):
            if modality == "img_attn":
                names = ["attn.to_q", "attn.to_k", "attn.to_v"]
            elif modality == "txt_attn":
                names = ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"]
            else:
                return []
            return [f"transformer_blocks.{block_index}.{name}" for name in names]
        mapped_name = _DOUBLE_BLOCK_LINEAR_MAP.get(within_block)
        if mapped_name:
            return [f"transformer_blocks.{block_index}.{mapped_name}"]

    if parts[0] == "single_blocks":
        block_index = parts[1]
        within_block = ".".join(parts[2:])
        mapped_name = _SINGLE_BLOCK_LINEAR_MAP.get(within_block)
        if mapped_name:
            return [f"single_transformer_blocks.{block_index}.{mapped_name}"]

    return []


def _install_scaled_fp8_linears(transformer, state_dict: dict[str, Any]) -> list[str]:
    import torch
    from torch import nn

    target_names = sorted({key.rsplit(".", maxsplit=1)[0] for key in state_dict if key.endswith(".weight_scale")})
    for name in target_names:
        module = _get_module(transformer, name)
        if not isinstance(module, nn.Linear):
            raise RuntimeError(f"FP8 target is not a Linear module: {name}")
        _set_module(transformer, name, _ScaledFP8Linear(torch, nn, module))
    return target_names


def _cast_non_fp8_parameters(module, dtype) -> None:
    for parameter in module.parameters():
        if parameter.dtype.is_floating_point and "float8" not in str(parameter.dtype):
            parameter.data = parameter.data.to(dtype=dtype)


def _get_module(root, name: str):
    module = root
    for part in name.split("."):
        module = getattr(module, part)
    return module


def _set_module(root, name: str, module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def _ScaledFP8Linear(torch, nn, source):
    class ScaledFP8Linear(nn.Linear):
        def __init__(self):
            super().__init__(
                source.in_features,
                source.out_features,
                bias=source.bias is not None,
                device=source.weight.device,
                dtype=torch.float32,
            )
            self.weight = nn.Parameter(
                torch.empty(
                    source.out_features,
                    source.in_features,
                    device=source.weight.device,
                    dtype=torch.float8_e4m3fn,
                ),
                requires_grad=False,
            )
            if source.bias is not None:
                self.bias = nn.Parameter(
                    torch.empty(source.out_features, device=source.weight.device, dtype=source.bias.dtype),
                    requires_grad=False,
                )
            self.register_buffer("weight_scale", torch.ones((), dtype=torch.float32))
            self.register_buffer("input_scale", torch.ones((), dtype=torch.float32))

        def forward(self, input):
            weight = self.weight.to(dtype=input.dtype)
            weight = weight * self.weight_scale.to(device=input.device, dtype=input.dtype)
            bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None
            return torch.nn.functional.linear(input, weight, bias)

        def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            for name in ("weight_scale", "input_scale"):
                key = f"{prefix}{name}"
                if key in state_dict:
                    setattr(self, name, state_dict.pop(key).to(dtype=torch.float32))
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name in ("weight_scale", "input_scale"):
                key = f"{prefix}{name}"
                if key in missing_keys:
                    missing_keys.remove(key)

    return ScaledFP8Linear()
