# -*- coding: utf-8 -*-
"""Qwen3 / Qwen3-MoE forward pass in pure JAX.

Parameters are stored in a flat dict keyed by their HuggingFace names.
Linear weights are transposed to ``(in_features, out_features)``; embeddings
and norm weights keep their HuggingFace shape.

The forward pass is a pure function of ``params`` so it can be wrapped with
``jax.jit`` and differentiated with ``jax.grad``.
"""
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from .config import Qwen3Config
from .lora import LoRAConfig, LoRAParams
from .ops import (
    apply_rotary_pos_emb,
    attention,
    causal_mask,
    linear,
    rms_norm,
    rope_cos_sin,
    silu,
)

Params = Dict[str, jnp.ndarray]
KVCache = Tuple[Tuple[jnp.ndarray, jnp.ndarray], ...]


# ---------------------------------------------------------------------------
# Parameter initialization
# ---------------------------------------------------------------------------

def _normal(rng, shape, std: float, dtype):
    return (jax.random.normal(rng, shape, dtype=jnp.float32) * std).astype(dtype)


def init_params(config: Qwen3Config, *, dtype=jnp.float32,
                rng: Optional[jnp.ndarray] = None) -> Params:
    """Create randomly initialized parameters matching the HF layout."""
    if rng is None:
        rng = jax.random.PRNGKey(0)
    hidden = config.hidden_size
    head_dim = config.head_dim
    q_out = config.num_attention_heads * head_dim
    kv_out = config.num_key_value_heads * head_dim
    std = config.initializer_range
    params: Params = {}

    def key():
        nonlocal rng
        rng, sub = jax.random.split(rng)
        return sub

    params["model.embed_tokens.weight"] = _normal(
        key(), (config.vocab_size, hidden), std, dtype)

    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        params[f"{p}.input_layernorm.weight"] = jnp.ones((hidden,), dtype)
        a = f"{p}.self_attn"
        params[f"{a}.q_proj.weight"] = _normal(key(), (hidden, q_out), std, dtype)
        params[f"{a}.k_proj.weight"] = _normal(key(), (hidden, kv_out), std, dtype)
        params[f"{a}.v_proj.weight"] = _normal(key(), (hidden, kv_out), std, dtype)
        params[f"{a}.o_proj.weight"] = _normal(key(), (q_out, hidden), std, dtype)
        if config.attention_bias:
            params[f"{a}.q_proj.bias"] = jnp.zeros((q_out,), dtype)
            params[f"{a}.k_proj.bias"] = jnp.zeros((kv_out,), dtype)
            params[f"{a}.v_proj.bias"] = jnp.zeros((kv_out,), dtype)
            params[f"{a}.o_proj.bias"] = jnp.zeros((hidden,), dtype)
        params[f"{a}.q_norm.weight"] = jnp.ones((head_dim,), dtype)
        params[f"{a}.k_norm.weight"] = jnp.ones((head_dim,), dtype)
        params[f"{p}.post_attention_layernorm.weight"] = jnp.ones((hidden,), dtype)

        m = f"{p}.mlp"
        if config.layer_is_moe(i):
            inter = config.moe_intermediate_size
            n_exp = config.num_experts
            params[f"{m}.gate.weight"] = _normal(key(), (n_exp, hidden), std, dtype)
            params[f"{m}.experts.gate_proj.weight"] = _normal(
                key(), (n_exp, hidden, inter), std, dtype)
            params[f"{m}.experts.up_proj.weight"] = _normal(
                key(), (n_exp, hidden, inter), std, dtype)
            params[f"{m}.experts.down_proj.weight"] = _normal(
                key(), (n_exp, inter, hidden), std, dtype)
        else:
            inter = config.intermediate_size
            params[f"{m}.gate_proj.weight"] = _normal(key(), (hidden, inter), std, dtype)
            params[f"{m}.up_proj.weight"] = _normal(key(), (hidden, inter), std, dtype)
            params[f"{m}.down_proj.weight"] = _normal(key(), (inter, hidden), std, dtype)

    params["model.norm.weight"] = jnp.ones((hidden,), dtype)
    if config.tie_word_embeddings:
        params["lm_head.weight"] = jnp.transpose(params["model.embed_tokens.weight"])
    else:
        params["lm_head.weight"] = _normal(
            key(), (hidden, config.vocab_size), std, dtype)
    return params


# ---------------------------------------------------------------------------
# LoRA plumbing
# ---------------------------------------------------------------------------

def _lora_pair(lora_params: Optional[LoRAParams], name: str
               ) -> Optional[Dict[str, jnp.ndarray]]:
    return None if lora_params is None else lora_params.get(name)


def _linear_lora(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    bias: Optional[jnp.ndarray],
    pair: Optional[Dict[str, jnp.ndarray]],
    scale: Optional[float],
) -> jnp.ndarray:
    """``x @ W + bias`` plus an optional low-rank update."""
    out = linear(x, weight, bias)
    if pair is not None:
        delta = jnp.dot(jnp.dot(x, pair["A"]), pair["B"])
        out = out + (scale * delta).astype(out.dtype)
    return out


def _expert_linear_lora(
    x_sorted: jnp.ndarray,
    weight: jnp.ndarray,
    pair: Optional[Dict[str, jnp.ndarray]],
    group_sizes: jnp.ndarray,
    scale: Optional[float],
) -> jnp.ndarray:
    """Grouped expert matmul with an optional per-expert low-rank update."""
    out = jax.lax.ragged_dot(x_sorted, weight, group_sizes)
    if pair is not None:
        projected = jax.lax.ragged_dot(x_sorted, pair["A"], group_sizes)
        delta = jax.lax.ragged_dot(projected, pair["B"], group_sizes)
        out = out + (scale * delta).astype(out.dtype)
    return out


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def _self_attention(
    config: Qwen3Config,
    params: Params,
    prefix: str,
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    mask: jnp.ndarray,
    past_kv: Optional[Tuple[jnp.ndarray, jnp.ndarray]],
    lora_params: Optional[LoRAParams] = None,
    lora_scale: Optional[float] = None,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    batch, seq, _ = x.shape
    head_dim = config.head_dim
    a = f"{prefix}.self_attn"

    q = _linear_lora(
        x, params[f"{a}.q_proj.weight"], params.get(f"{a}.q_proj.bias"),
        _lora_pair(lora_params, f"{a}.q_proj.weight"), lora_scale)
    k = _linear_lora(
        x, params[f"{a}.k_proj.weight"], params.get(f"{a}.k_proj.bias"),
        _lora_pair(lora_params, f"{a}.k_proj.weight"), lora_scale)
    v = _linear_lora(
        x, params[f"{a}.v_proj.weight"], params.get(f"{a}.v_proj.bias"),
        _lora_pair(lora_params, f"{a}.v_proj.weight"), lora_scale)

    q = q.reshape(batch, seq, config.num_attention_heads, head_dim)
    k = k.reshape(batch, seq, config.num_key_value_heads, head_dim)
    v = v.reshape(batch, seq, config.num_key_value_heads, head_dim)

    q = rms_norm(q, params[f"{a}.q_norm.weight"], config.rms_norm_eps)
    k = rms_norm(k, params[f"{a}.k_norm.weight"], config.rms_norm_eps)

    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k = jnp.swapaxes(k, 1, 2)  # (B, Hkv, S, D)
    v = jnp.swapaxes(v, 1, 2)

    if past_kv is not None:
        k = jnp.concatenate([past_kv[0], k], axis=2)
        v = jnp.concatenate([past_kv[1], v], axis=2)

    scale = head_dim ** -0.5
    out = attention(q, k, v, mask, config.num_key_value_groups, scale)
    out = _linear_lora(
        out, params[f"{a}.o_proj.weight"], params.get(f"{a}.o_proj.bias"),
        _lora_pair(lora_params, f"{a}.o_proj.weight"), lora_scale)
    return out, (k, v)


def _dense_mlp(config: Qwen3Config, params: Params, prefix: str,
               x: jnp.ndarray,
               lora_params: Optional[LoRAParams] = None,
               lora_scale: Optional[float] = None) -> jnp.ndarray:
    m = f"{prefix}.mlp"
    gate = _linear_lora(
        x, params[f"{m}.gate_proj.weight"], None,
        _lora_pair(lora_params, f"{m}.gate_proj.weight"), lora_scale)
    up = _linear_lora(
        x, params[f"{m}.up_proj.weight"], None,
        _lora_pair(lora_params, f"{m}.up_proj.weight"), lora_scale)
    return _linear_lora(
        silu(gate) * up, params[f"{m}.down_proj.weight"], None,
        _lora_pair(lora_params, f"{m}.down_proj.weight"), lora_scale)


def _moe_mlp(config: Qwen3Config, params: Params, prefix: str,
             x: jnp.ndarray,
             lora_params: Optional[LoRAParams] = None,
             lora_scale: Optional[float] = None
             ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Dropless top-k MoE using ``lax.ragged_dot`` for grouped matmuls."""
    m = f"{prefix}.mlp"
    orig_shape = x.shape
    hidden = orig_shape[-1]
    x_flat = x.reshape(-1, hidden)
    n_tokens = x_flat.shape[0]
    k = config.num_experts_per_tok
    n_exp = config.num_experts

    router_logits = jnp.dot(x_flat, jnp.transpose(params[f"{m}.gate.weight"]))
    probs = jax.nn.softmax(router_logits.astype(jnp.float32), axis=-1)
    top_w, top_idx = jax.lax.top_k(probs, k)
    if config.norm_topk_prob:
        top_w = top_w / jnp.sum(top_w, axis=-1, keepdims=True)
    top_w = top_w.astype(x.dtype)

    expert_ids = top_idx.reshape(-1)                       # (N*k,)
    token_ids = jnp.repeat(jnp.arange(n_tokens), k)        # (N*k,)
    flat_w = top_w.reshape(-1)                             # (N*k,)

    order = jnp.argsort(expert_ids)
    sorted_expert = expert_ids[order]
    sorted_token = token_ids[order]
    sorted_w = flat_w[order]
    group_sizes = jax.ops.segment_sum(
        jnp.ones_like(sorted_expert), sorted_expert, num_segments=n_exp)

    x_sorted = x_flat[sorted_token]                        # (N*k, H)
    gate_w = params[f"{m}.experts.gate_proj.weight"]       # (E, H, I)
    up_w = params[f"{m}.experts.up_proj.weight"]
    down_w = params[f"{m}.experts.down_proj.weight"]       # (E, I, H)

    gate = _expert_linear_lora(
        x_sorted, gate_w,
        _lora_pair(lora_params, f"{m}.experts.gate_proj.weight"),
        group_sizes, lora_scale)
    up = _expert_linear_lora(
        x_sorted, up_w,
        _lora_pair(lora_params, f"{m}.experts.up_proj.weight"),
        group_sizes, lora_scale)
    expert_out = _expert_linear_lora(
        silu(gate) * up, down_w,
        _lora_pair(lora_params, f"{m}.experts.down_proj.weight"),
        group_sizes, lora_scale)

    weighted = expert_out * sorted_w[:, None]
    out = jax.ops.segment_sum(weighted, sorted_token, num_segments=n_tokens)
    return out.reshape(orig_shape), router_logits.reshape(*orig_shape[:-1], n_exp)


def _decoder_layer(
    config: Qwen3Config,
    params: Params,
    layer_idx: int,
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    mask: jnp.ndarray,
    past_kv: Optional[Tuple[jnp.ndarray, jnp.ndarray]],
    collect_router_logits: bool,
    lora_params: Optional[LoRAParams] = None,
    lora_scale: Optional[float] = None,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray], Optional[jnp.ndarray]]:
    prefix = f"model.layers.{layer_idx}"

    residual = x
    h = rms_norm(x, params[f"{prefix}.input_layernorm.weight"], config.rms_norm_eps)
    attn_out, new_kv = _self_attention(
        config, params, prefix, h, cos, sin, mask, past_kv,
        lora_params, lora_scale)
    h = residual + attn_out

    residual = h
    h = rms_norm(h, params[f"{prefix}.post_attention_layernorm.weight"],
                 config.rms_norm_eps)
    if config.layer_is_moe(layer_idx):
        mlp_out, router_logits = _moe_mlp(
            config, params, prefix, h, lora_params, lora_scale)
        if not collect_router_logits:
            router_logits = None
    else:
        mlp_out = _dense_mlp(config, params, prefix, h, lora_params, lora_scale)
        router_logits = None

    return residual + mlp_out, new_kv, router_logits


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward(
    config: Qwen3Config,
    params: Params,
    input_ids: jnp.ndarray,
    attention_mask: Optional[jnp.ndarray] = None,
    position_ids: Optional[jnp.ndarray] = None,
    past_key_values: Optional[KVCache] = None,
    use_cache: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    lora_params: Optional[LoRAParams] = None,
    lora_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the decoder and return logits, KV cache and optional router logits.

    ``attention_mask`` is expected to cover the whole cached sequence
    (shape ``(B, past_len + T)``).  ``position_ids`` are absolute positions.
    ``lora_params`` adds low-rank updates to the targeted linear projections.
    """
    if use_cache is None:
        use_cache = config.use_cache
    if output_router_logits is None:
        output_router_logits = config.output_router_logits

    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    batch, seq = input_ids.shape
    past_len = 0
    if past_key_values is not None:
        past_len = past_key_values[0][0].shape[2]

    if position_ids is None:
        position_ids = jnp.arange(past_len, past_len + seq, dtype=jnp.int32)
    else:
        position_ids = jnp.asarray(position_ids, dtype=jnp.int32)

    kv_len = past_len + seq
    if attention_mask is None:
        attention_mask = jnp.ones((batch, kv_len), dtype=jnp.int32)
    mask = causal_mask(seq, kv_len, past_len, attention_mask)

    cos, sin = rope_cos_sin(config.head_dim, position_ids, config.rope_theta)

    h = params["model.embed_tokens.weight"][input_ids]
    new_caches: List[Tuple[jnp.ndarray, jnp.ndarray]] = []
    router_logits_list: List[jnp.ndarray] = []

    for i in range(config.num_hidden_layers):
        past = past_key_values[i] if past_key_values is not None else None
        h, kv, router_logits = _decoder_layer(
            config, params, i, h, cos, sin, mask, past,
            output_router_logits, lora_params, lora_scale)
        new_caches.append(kv)
        if router_logits is not None:
            router_logits_list.append(
                router_logits.reshape(-1, config.num_experts))

    h = rms_norm(h, params["model.norm.weight"], config.rms_norm_eps)
    logits = jnp.dot(h, params["lm_head.weight"]).astype(jnp.float32)

    aux_loss = jnp.asarray(0.0, dtype=jnp.float32)
    if router_logits_list:
        aux_loss = load_balancing_loss(
            router_logits_list, config.num_experts_per_tok,
            config.num_experts, attention_mask[:, past_len:])

    return {
        "logits": logits,
        "past_key_values": tuple(new_caches) if use_cache else None,
        "aux_loss": aux_loss,
        "router_logits": tuple(router_logits_list) if output_router_logits else None,
    }


def load_balancing_loss(
    router_logits: List[jnp.ndarray],
    top_k: int,
    num_experts: int,
    attention_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Switch-transformer style auxiliary loss (matches HF Qwen3-MoE)."""
    stacked = jnp.stack(router_logits, axis=0)  # (L, N, E)
    n_layers, n_tokens = stacked.shape[:2]
    routing_weights = jax.nn.softmax(stacked.astype(jnp.float32), axis=-1)
    selected = jax.lax.top_k(routing_weights, top_k)[1]
    expert_mask = jax.nn.one_hot(selected, num_experts, dtype=jnp.float32)

    if attention_mask is None:
        tokens_sum = jnp.sum(expert_mask, axis=(0, 1, 2))
        prob_sum = jnp.sum(routing_weights, axis=(0, 1))
        total_rows = n_layers * n_tokens
    else:
        flat_mask = attention_mask.reshape(-1).astype(jnp.float32)
        mask_k = flat_mask[None, :, None, None]
        tokens_sum = jnp.sum(expert_mask * mask_k, axis=(0, 1, 2))
        prob_sum = jnp.sum(
            routing_weights * flat_mask[None, :, None], axis=(0, 1))
        total_rows = n_layers * jnp.sum(flat_mask)

    total_rows = jnp.maximum(total_rows, 1.0)
    tokens_per_expert = tokens_sum / total_rows
    router_prob_per_expert = prob_sum / total_rows

    overall = jnp.sum(tokens_per_expert * router_prob_per_expert)
    return overall * num_experts


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _apply_top_k(logits: jnp.ndarray, top_k: int) -> jnp.ndarray:
    k = min(top_k, logits.shape[-1])
    top_vals = jax.lax.top_k(logits, k)[0]
    threshold = top_vals[..., -1:]
    return jnp.where(logits >= threshold, logits, -jnp.inf)


def _apply_top_p(logits: jnp.ndarray, top_p: float) -> jnp.ndarray:
    sorted_idx = jnp.argsort(-logits, axis=-1)
    sorted_logits = jnp.take_along_axis(logits, sorted_idx, axis=-1)
    probs = jax.nn.softmax(sorted_logits, axis=-1)
    cumulative = jnp.cumsum(probs, axis=-1)
    remove = (cumulative - probs) > top_p
    sorted_logits = jnp.where(remove, -jnp.inf, sorted_logits)
    out = jnp.full_like(logits, -jnp.inf)
    return jnp.put_along_axis(out, sorted_idx, sorted_logits, axis=-1,
                              inplace=False)


def _sample(logits: jnp.ndarray, rng: jnp.ndarray, temperature: float,
            top_k: Optional[int], top_p: Optional[float]) -> jnp.ndarray:
    if temperature is None or temperature <= 0:
        return jnp.argmax(logits, axis=-1)
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        logits = _apply_top_k(logits, top_k)
    if top_p is not None and 0 < top_p < 1.0:
        logits = _apply_top_p(logits, top_p)
    return jax.random.categorical(rng, logits, axis=-1)


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class _Qwen3CausalLMBase:
    """Shared implementation for dense and MoE Qwen3 causal LMs."""

    def __init__(self, config: Qwen3Config, params: Optional[Params] = None, *,
                 dtype=jnp.float32, rng: Optional[jnp.ndarray] = None):
        self.config = config
        self.params = params if params is not None else init_params(
            config, dtype=dtype, rng=rng)
        self.lora_params: Optional[LoRAParams] = None
        self.lora_config: Optional[LoRAConfig] = None

    # -- LoRA ---------------------------------------------------------------
    def add_lora(self, rank: int = 8, alpha: float = 16.0,
                 target_modules: Optional[Tuple[str, ...]] = None,
                 init_std: float = 0.01,
                 rng: Optional[jnp.ndarray] = None) -> None:
        """Attach low-rank adapters; subsequent ``fit`` trains only these."""
        from .lora import init_lora_params
        self.lora_config = LoRAConfig(
            rank=rank, alpha=alpha, init_std=init_std,
            target_modules=(target_modules if target_modules is not None
                            else LoRAConfig().target_modules))
        self.lora_params = init_lora_params(
            self.params, self.lora_config, rng=rng)

    def merge_lora(self) -> None:
        """Merge adapter weights into the base parameters and detach them."""
        if self.lora_params is None:
            raise ValueError("尚未调用 add_lora")
        from .lora import merge_lora_params
        self.params = merge_lora_params(
            self.params, self.lora_params, self.lora_config.scale)
        self.lora_params = None
        self.lora_config = None

    def _lora_scale(self) -> Optional[float]:
        if self.lora_params is None or self.lora_config is None:
            return None
        return self.lora_config.scale

    # -- construction -------------------------------------------------------
    @classmethod
    def from_pretrained(cls, path_or_repo_id: str, *, endpoint: Optional[str] = None,
                        revision: Optional[str] = None, cache_dir: Optional[str] = None,
                        dtype: Optional[str] = None):
        from .loading import load_pretrained
        config, params = load_pretrained(
            path_or_repo_id, endpoint=endpoint, revision=revision,
            cache_dir=cache_dir, dtype=dtype)
        return cls(config, params)

    def save_pretrained(self, save_directory: str, dtype: str = "float32") -> None:
        from .loading import save_pretrained
        save_pretrained(self, save_directory, dtype=dtype)

    # -- forward ------------------------------------------------------------
    def __call__(self, input_ids, attention_mask=None, position_ids=None,
                 past_key_values=None, use_cache: Optional[bool] = None,
                 output_router_logits: Optional[bool] = None):
        return forward(
            self.config, self.params, input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            lora_params=self.lora_params,
            lora_scale=self._lora_scale(),
        )

    # -- generation ---------------------------------------------------------
    def generate(self, input_ids, max_new_tokens: int = 32,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 eos_token_id: Optional[int] = None,
                 attention_mask=None, rng: Optional[jnp.ndarray] = None,
                 use_cache: bool = True):
        """Autoregressive generation; returns prompt + continuation ids."""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        batch = input_ids.shape[0]
        if attention_mask is None:
            attention_mask = jnp.ones(input_ids.shape, dtype=jnp.int32)
        else:
            attention_mask = jnp.asarray(attention_mask, dtype=jnp.int32)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask[None, :]

        pieces = [input_ids]
        out = self(input_ids, attention_mask, use_cache=use_cache)
        finished = jnp.zeros((batch,), dtype=bool)

        for _ in range(max_new_tokens):
            rng, step_rng = jax.random.split(rng)
            next_ids = _sample(out["logits"][:, -1, :], step_rng, temperature,
                               top_k, top_p)
            next_ids = next_ids.astype(jnp.int32)
            pieces.append(next_ids[:, None])
            if eos_token_id is not None:
                finished = jnp.logical_or(finished, next_ids == eos_token_id)
                if bool(jnp.all(finished)):
                    break
            attention_mask = jnp.concatenate(
                [attention_mask, jnp.ones((batch, 1), dtype=jnp.int32)], axis=1)
            if use_cache:
                step_input = next_ids[:, None]
                out = self(step_input, attention_mask,
                           past_key_values=out["past_key_values"],
                           use_cache=True)
            else:
                out = self(jnp.concatenate(pieces, axis=1), attention_mask,
                           use_cache=False)

        return jnp.concatenate(pieces, axis=1)

    # -- training -----------------------------------------------------------
    def loss(self, input_ids, attention_mask=None, labels=None):
        from .train import causal_lm_loss
        return causal_lm_loss(
            self.config, self.params, input_ids,
            attention_mask=attention_mask, labels=labels,
            lora_params=self.lora_params, lora_scale=self._lora_scale())

    def fit(self, batches, epochs: int = 1, lr: float = 3e-4,
            weight_decay: float = 0.1, grad_accum: int = 1,
            clip_norm: Optional[float] = 1.0, verbose: bool = False):
        from .train import fit
        use_lora = self.lora_params is not None
        updated, history = fit(
            self, batches, epochs=epochs, lr=lr,
            weight_decay=weight_decay, grad_accum=grad_accum,
            clip_norm=clip_norm, verbose=verbose)
        if use_lora:
            self.lora_params = updated
        else:
            self.params = updated
        return history


class Qwen3ForCausalLM(_Qwen3CausalLMBase):
    """Dense Qwen3 causal language model."""


class Qwen3MoeForCausalLM(_Qwen3CausalLMBase):
    """Mixture-of-Experts Qwen3 causal language model."""
