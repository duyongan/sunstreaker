# -*- coding: utf-8 -*-
"""Core Qwen3 math in pure JAX.

All helpers are stateless functions so they can be jitted and differentiated.
Parameter arrays are stored transposed relative to HuggingFace Linear weights:
``(in_features, out_features)``.
"""
from typing import Optional, Tuple

import jax
import jax.numpy as jnp


def linear(x: jnp.ndarray, weight: jnp.ndarray, bias: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """``x @ weight (+ bias)`` with weight shaped ``(in, out)``."""
    out = jnp.dot(x, weight)
    if bias is not None:
        out = out + bias
    return out


def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float) -> jnp.ndarray:
    """RMSNorm; reductions are done in float32 for stability."""
    dtype = x.dtype
    x_f = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x_f), axis=-1, keepdims=True)
    y = x_f * jax.lax.rsqrt(var + eps)
    return (y * weight.astype(jnp.float32)).astype(dtype)


def silu(x: jnp.ndarray) -> jnp.ndarray:
    return x * jax.nn.sigmoid(x)


def swiglu(x: jnp.ndarray, gate_w: jnp.ndarray, up_w: jnp.ndarray,
           down_w: jnp.ndarray) -> jnp.ndarray:
    gate = linear(x, gate_w)
    up = linear(x, up_w)
    return linear(silu(gate) * up, down_w)


def rope_cos_sin(
    head_dim: int,
    position_ids: jnp.ndarray,
    theta: float,
    dtype=jnp.float32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Standard split-half RoPE tables shaped ``(T, head_dim)``."""
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = jnp.asarray(position_ids, dtype=jnp.float32)[:, None] * inv_freq[None, :]
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    half = x.shape[-1] // 2
    return jnp.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def apply_rotary_pos_emb(
    q: jnp.ndarray,
    k: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply RoPE to q/k of shape ``(B, T, H, D)``."""
    cos = cos[None, :, None, :].astype(q.dtype)
    sin = sin[None, :, None, :].astype(q.dtype)
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out, k_out


def repeat_kv(x: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    """Repeat KV heads: ``(B, Hkv, S, D) -> (B, Hkv*n_rep, S, D)``."""
    if n_rep == 1:
        return x
    batch, heads, seq, dim = x.shape
    x = jnp.broadcast_to(x[:, :, None, :, :], (batch, heads, n_rep, seq, dim))
    return x.reshape(batch, heads * n_rep, seq, dim)


def causal_mask(
    q_len: int,
    kv_len: int,
    past_len: int = 0,
    attention_mask: Optional[jnp.ndarray] = None,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Boolean mask ``(B or 1, 1, q_len, kv_len)``; True means attend."""
    q_pos = past_len + jnp.arange(q_len)
    kv_pos = jnp.arange(kv_len)
    mask = kv_pos[None, :] <= q_pos[:, None]  # (q, kv)
    mask = mask[None, None, :, :]
    if attention_mask is not None:
        mask = jnp.logical_and(mask, attention_mask[:, None, None, :].astype(bool))
    return jnp.broadcast_to(mask, mask.shape)


def attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    mask: jnp.ndarray,
    num_kv_groups: int,
    scale: float,
) -> jnp.ndarray:
    """Scaled dot-product attention with GQA.

    q: ``(B, T, Hq, D)``, k/v: ``(B, Hkv, S, D)``, mask: broadcastable to
    ``(B, Hq, T, S)``. Returns ``(B, T, Hq*D)``.
    """
    k = repeat_kv(k, num_kv_groups)  # (B, Hq, S, D)
    v = repeat_kv(v, num_kv_groups)
    q = jnp.swapaxes(q, 1, 2)  # (B, Hq, T, D)

    scores = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale
    scores = scores.astype(jnp.float32)
    scores = jnp.where(mask, scores, -1e9)
    probs = jax.nn.softmax(scores, axis=-1).astype(q.dtype)

    out = jnp.einsum("bhts,bhsd->bhtd", probs, v)
    out = jnp.swapaxes(out, 1, 2)  # (B, T, Hq, D)
    return out.reshape(out.shape[0], out.shape[1], -1)
