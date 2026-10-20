# -*- coding: utf-8 -*-
"""Loading and saving Qwen3 safetensors checkpoints.

HuggingFace stores linear weights as ``(out_features, in_features)`` while the
JAX model uses ``(in_features, out_features)``.  This module converts between
the two layouts and stacks per-expert MoE weights into ``(num_experts, ...)``
tensors.
"""
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from .config import Qwen3Config

CONFIG_NAME = "config.json"
INDEX_NAME = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
SUPPORTED_TYPES = ("qwen3", "qwen3_moe")

_EXPERT_RE = re.compile(
    r"^(model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
# transformers 5.x stores all experts in fused tensors.
_FUSED_GATE_UP_RE = re.compile(
    r"^(model\.layers\.\d+\.mlp)\.experts\.gate_up_proj$")
_FUSED_DOWN_RE = re.compile(
    r"^(model\.layers\.\d+\.mlp)\.experts\.down_proj$")
_STACKED_EXPERT_RE = re.compile(
    r"^(model\.layers\.\d+\.mlp)\.experts\.(gate_proj|up_proj|down_proj)$")

# Linear projections stored by HF as (out, in).
_TRANSPOSED_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_model_dir(path_or_repo_id: str, *, endpoint: Optional[str] = None,
                      revision: Optional[str] = None,
                      cache_dir: Optional[str] = None) -> str:
    """Return a local directory for either a local path or a Hub repo id."""
    if os.path.isdir(path_or_repo_id):
        return path_or_repo_id
    from huggingface_hub import snapshot_download

    endpoint = endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT
    return snapshot_download(
        repo_id=path_or_repo_id,
        revision=revision,
        cache_dir=cache_dir,
        endpoint=endpoint,
        allow_patterns=[
            "*.json", "*.safetensors", "*.txt", "*.model", "*.jinja",
            "tokenizer*", "*.tiktoken", "merges.txt", "vocab.*",
        ],
    )


# ---------------------------------------------------------------------------
# safetensors reading (with a tiny BF16-safe fallback parser)
# ---------------------------------------------------------------------------

def _read_safetensors_manual(path: str) -> Dict[str, np.ndarray]:
    """Minimal safetensors reader used when the installed library cannot
    decode BF16 with its numpy backend."""
    import ml_dtypes

    numpy_dtypes = {
        "F64": np.float64,
        "F32": np.float32,
        "F16": np.float16,
        "BF16": ml_dtypes.bfloat16,
        "I64": np.int64,
        "I32": np.int32,
        "I16": np.int16,
        "I8": np.int8,
        "U8": np.uint8,
        "BOOL": np.bool_,
    }
    tensors: Dict[str, np.ndarray] = {}
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_start = 8 + header_len
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = info["dtype"]
            if dtype not in numpy_dtypes:
                raise ValueError(f"不支持的 safetensors dtype: {dtype} ({name})")
            begin, end = info["data_offsets"]
            f.seek(data_start + begin)
            raw = f.read(end - begin)
            arr = np.frombuffer(raw, dtype=numpy_dtypes[dtype])
            tensors[name] = arr.reshape(info["shape"])
    return tensors


def _read_shard(path: str, names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    try:
        from safetensors import safe_open

        out = {}
        with safe_open(path, framework="numpy") as f:
            available = set(f.keys())
            for name in (names if names is not None else sorted(available)):
                if name in available:
                    out[name] = f.get_tensor(name)
        return out
    except Exception:
        all_tensors = _read_safetensors_manual(path)
        if names is None:
            return all_tensors
        return {n: all_tensors[n] for n in names if n in all_tensors}


def _load_checkpoint_tensors(model_dir: str) -> Dict[str, np.ndarray]:
    """Read every tensor, honouring sharded ``index.json`` files."""
    index_path = os.path.join(model_dir, INDEX_NAME)
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        by_shard: Dict[str, List[str]] = {}
        for name, shard in weight_map.items():
            by_shard.setdefault(shard, []).append(name)
        tensors: Dict[str, np.ndarray] = {}
        for shard, names in by_shard.items():
            shard_path = os.path.join(model_dir, shard)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"缺少权重分片: {shard_path}")
            tensors.update(_read_shard(shard_path, names))
        return tensors

    single = os.path.join(model_dir, SINGLE_FILE)
    if os.path.exists(single):
        return _read_shard(single)

    candidates = sorted(
        f for f in os.listdir(model_dir)
        if f.endswith(".safetensors") and not f.startswith(".")
    )
    if len(candidates) == 1:
        return _read_shard(os.path.join(model_dir, candidates[0]))
    if not candidates:
        raise FileNotFoundError(
            f"{model_dir} 中没有找到 *.safetensors 权重文件")
    raise FileNotFoundError(
        f"{model_dir} 有多个 safetensors 文件但缺少 {INDEX_NAME}")


# ---------------------------------------------------------------------------
# Weight mapping
# ---------------------------------------------------------------------------

def _transpose(name: str) -> bool:
    if name == "lm_head.weight":
        return True
    return any(name.endswith(s) for s in _TRANSPOSED_SUFFIXES)


def _map_checkpoint(
    tensors: Dict[str, np.ndarray],
    config: Qwen3Config,
    dtype: jnp.dtype,
) -> Dict[str, jnp.ndarray]:
    params: Dict[str, jnp.ndarray] = {}
    experts: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}

    for name, value in tensors.items():
        if "rotary_emb.inv_freq" in name:
            continue
        match = _EXPERT_RE.match(name)
        if match:
            prefix, expert_idx, proj = match.groups()
            experts.setdefault(prefix, {}).setdefault(int(expert_idx), {})[proj] = value
            continue
        match = _FUSED_GATE_UP_RE.match(name)
        if match:
            prefix = match.group(1)
            inter = config.moe_intermediate_size
            arr = np.swapaxes(value, 1, 2)  # (E, 2I, H) -> (E, H, 2I)
            params[f"{prefix}.experts.gate_proj.weight"] = jnp.asarray(
                arr[..., :inter]).astype(dtype)
            params[f"{prefix}.experts.up_proj.weight"] = jnp.asarray(
                arr[..., inter:]).astype(dtype)
            continue
        match = _FUSED_DOWN_RE.match(name)
        if match:
            prefix = match.group(1)
            arr = np.swapaxes(value, 1, 2)  # (E, H, I) -> (E, I, H)
            params[f"{prefix}.experts.down_proj.weight"] = jnp.asarray(
                arr).astype(dtype)
            continue
        match = _STACKED_EXPERT_RE.match(name)
        if match:
            prefix, proj = match.groups()
            arr = np.swapaxes(value, 1, 2)
            params[f"{prefix}.experts.{proj}.weight"] = jnp.asarray(arr).astype(dtype)
            continue
        if ".rotary_emb." in name:
            continue
        arr = value
        if _transpose(name):
            arr = np.transpose(arr)
        params[name] = jnp.asarray(arr).astype(dtype)

    for prefix, per_expert in experts.items():
        indices = sorted(per_expert)
        if indices != list(range(len(indices))):
            raise ValueError(f"{prefix}.experts 的专家编号不连续: {indices[:10]}")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            stacked = np.stack([np.transpose(per_expert[i][proj]) for i in indices])
            params[f"{prefix}.experts.{proj}.weight"] = jnp.asarray(stacked).astype(dtype)

    return params


def _param_shapes(config: Qwen3Config) -> Dict[str, Tuple[int, ...]]:
    hidden = config.hidden_size
    head_dim = config.head_dim
    q_out = config.num_attention_heads * head_dim
    kv_out = config.num_key_value_heads * head_dim
    shapes: Dict[str, Tuple[int, ...]] = {
        "model.embed_tokens.weight": (config.vocab_size, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (hidden, config.vocab_size),
    }
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        a = f"{p}.self_attn"
        shapes[f"{p}.input_layernorm.weight"] = (hidden,)
        shapes[f"{p}.post_attention_layernorm.weight"] = (hidden,)
        shapes[f"{a}.q_proj.weight"] = (hidden, q_out)
        shapes[f"{a}.k_proj.weight"] = (hidden, kv_out)
        shapes[f"{a}.v_proj.weight"] = (hidden, kv_out)
        shapes[f"{a}.o_proj.weight"] = (q_out, hidden)
        shapes[f"{a}.q_norm.weight"] = (head_dim,)
        shapes[f"{a}.k_norm.weight"] = (head_dim,)
        if config.attention_bias:
            shapes[f"{a}.q_proj.bias"] = (q_out,)
            shapes[f"{a}.k_proj.bias"] = (kv_out,)
            shapes[f"{a}.v_proj.bias"] = (kv_out,)
            shapes[f"{a}.o_proj.bias"] = (hidden,)
        m = f"{p}.mlp"
        if config.layer_is_moe(i):
            shapes[f"{m}.gate.weight"] = (config.num_experts, hidden)
            shapes[f"{m}.experts.gate_proj.weight"] = (
                config.num_experts, hidden, config.moe_intermediate_size)
            shapes[f"{m}.experts.up_proj.weight"] = (
                config.num_experts, hidden, config.moe_intermediate_size)
            shapes[f"{m}.experts.down_proj.weight"] = (
                config.num_experts, config.moe_intermediate_size, hidden)
        else:
            shapes[f"{m}.gate_proj.weight"] = (hidden, config.intermediate_size)
            shapes[f"{m}.up_proj.weight"] = (hidden, config.intermediate_size)
            shapes[f"{m}.down_proj.weight"] = (config.intermediate_size, hidden)
    return shapes


def _resolve_dtype(config: Qwen3Config, dtype: Optional[str]) -> jnp.dtype:
    candidate = dtype or config.torch_dtype or "float32"
    if candidate in ("bfloat16", "bf16"):
        return jnp.bfloat16
    if candidate in ("float16", "fp16", "half"):
        return jnp.float16
    if candidate == "float64":
        return jnp.float64
    return jnp.float32


def load_pretrained(path_or_repo_id: str, *, endpoint: Optional[str] = None,
                    revision: Optional[str] = None, cache_dir: Optional[str] = None,
                    dtype: Optional[str] = None
                    ) -> Tuple[Qwen3Config, Dict[str, jnp.ndarray]]:
    """Load config and parameters from a local directory or Hub repo id."""
    model_dir = resolve_model_dir(
        path_or_repo_id, endpoint=endpoint, revision=revision, cache_dir=cache_dir)
    config = Qwen3Config.from_json_file(os.path.join(model_dir, CONFIG_NAME))
    if config.model_type not in SUPPORTED_TYPES:
        raise ValueError(
            f"model_type={config.model_type!r} 不是受支持的 Qwen3 配置")

    jnp_dtype = _resolve_dtype(config, dtype)
    tensors = _load_checkpoint_tensors(model_dir)
    params = _map_checkpoint(tensors, config, jnp_dtype)
    del tensors

    if config.tie_word_embeddings:
        params["lm_head.weight"] = jnp.transpose(params["model.embed_tokens.weight"])

    expected = _param_shapes(config)
    missing = sorted(set(expected) - set(params))
    extra = sorted(set(params) - set(expected))
    if missing:
        raise ValueError(
            "权重缺失以下参数（可能与 config 不匹配）:\n  " + "\n  ".join(missing[:10])
            + ("" if len(missing) <= 10 else f"\n  ... 共 {len(missing)} 个"))
    if extra:
        raise ValueError(
            "权重包含未知参数:\n  " + "\n  ".join(extra[:10])
            + ("" if len(extra) <= 10 else f"\n  ... 共 {len(extra)} 个"))
    for name, shape in expected.items():
        if tuple(params[name].shape) != tuple(shape):
            raise ValueError(
                f"参数 {name} 形状不匹配: 期望 {shape}, 实际 {tuple(params[name].shape)}")

    return config, params


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def _to_numpy(array, dtype: np.dtype) -> np.ndarray:
    return np.asarray(jnp.asarray(array).astype(jnp.dtype(dtype)))


def save_pretrained(model, save_directory: str, dtype: str = "float32") -> None:
    """Write ``config.json`` and ``model.safetensors`` in HF layout."""
    from safetensors.numpy import save_file

    os.makedirs(save_directory, exist_ok=True)
    model.config.save_pretrained(save_directory)

    np_dtype = np.dtype(dtype)
    arrays: Dict[str, np.ndarray] = {}
    for name, value in model.params.items():
        if name == "lm_head.weight" and model.config.tie_word_embeddings:
            continue
        match = re.match(
            r"^(model\.layers\.\d+\.mlp)\.experts\.(gate_proj|up_proj|down_proj)\.weight$",
            name)
        if match:
            prefix, proj = match.groups()
            stacked = _to_numpy(value, np_dtype)
            for idx in range(stacked.shape[0]):
                arrays[f"{prefix}.experts.{idx}.{proj}.weight"] = np.ascontiguousarray(
                    np.transpose(stacked[idx]))
            continue
        arr = _to_numpy(value, np_dtype)
        if _transpose(name):
            # Attention/dense-MLP projections live in ``(in, out)`` order
            # locally; HF expects ``(out, in)``.
            arr = np.ascontiguousarray(np.transpose(arr))
        arrays[name] = arr

    save_file(arrays, os.path.join(save_directory, SINGLE_FILE))
