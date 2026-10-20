# -*- coding: utf-8 -*-
import numpy as np
import jax
import jax.numpy as jnp

from sunstreaker.applications.transformers.qwen3.ops import (
    apply_rotary_pos_emb,
    attention,
    causal_mask,
    repeat_kv,
    rms_norm,
    rope_cos_sin,
    silu,
    swiglu,
)


def test_rms_norm_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 8)).astype(np.float32)
    w = rng.standard_normal(8).astype(np.float32)
    expected = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6) * w
    actual = np.asarray(rms_norm(jnp.asarray(x), jnp.asarray(w), 1e-6))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_rope_tables_and_rotation():
    head_dim = 8
    positions = jnp.arange(3)
    cos, sin = rope_cos_sin(head_dim, positions, 10000.0)
    inv = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2) / head_dim))
    freqs = positions[:, None] * inv[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    np.testing.assert_allclose(np.asarray(cos), np.cos(emb), atol=1e-6)
    np.testing.assert_allclose(np.asarray(sin), np.sin(emb), atol=1e-6)

    x = jnp.asarray(np.random.default_rng(1).standard_normal((1, 3, 2, head_dim)),
                    dtype=jnp.float32)
    q, k = apply_rotary_pos_emb(x, x, cos, sin)
    half = head_dim // 2
    rotated = np.concatenate([-np.asarray(x)[..., half:], np.asarray(x)[..., :half]],
                             axis=-1)
    expected = np.asarray(x) * np.asarray(cos)[None, :, None, :] + \
        rotated * np.asarray(sin)[None, :, None, :]
    np.testing.assert_allclose(np.asarray(q), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(k), expected, rtol=1e-6, atol=1e-6)


def test_repeat_kv():
    x = jnp.arange(2 * 2 * 3 * 4, dtype=jnp.float32).reshape(2, 2, 3, 4)
    out = repeat_kv(x, 3)
    assert out.shape == (2, 6, 3, 4)
    np.testing.assert_array_equal(np.asarray(out[:, 0]), np.asarray(x[:, 0]))
    np.testing.assert_array_equal(np.asarray(out[:, 2]), np.asarray(x[:, 0]))
    np.testing.assert_array_equal(np.asarray(out[:, 3]), np.asarray(x[:, 1]))


def test_causal_mask_with_past():
    mask = np.asarray(causal_mask(2, 4, past_len=2, attention_mask=None))
    # query positions 2, 3 may attend to keys 0..2 and 0..3 respectively.
    expected = np.array([
        [True, True, True, False],
        [True, True, True, True],
    ])
    np.testing.assert_array_equal(mask[0, 0], expected)

    padded = np.asarray(causal_mask(
        2, 4, past_len=2,
        attention_mask=jnp.array([[1, 1, 0, 1]])))
    np.testing.assert_array_equal(
        padded[0, 0],
        np.array([[True, True, False, False],
                  [True, True, False, True]]))


def _reference_attention(q, k, v, mask, groups, scale):
    batch, q_len, q_heads, dim = q.shape
    mask = np.broadcast_to(mask, (batch, q_heads, q_len, k.shape[2]))
    out = np.zeros((batch, q_len, q_heads, dim), dtype=np.float32)
    for b in range(batch):
        for h in range(q_heads):
            kv = k[b, h // groups]
            scores = q[b, :, h] @ kv.T * scale
            scores = np.where(mask[b, h], scores, -1e9)
            probs = np.exp(scores - scores.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            out[b, :, h] = probs @ v[b, h // groups]
    return out.reshape(batch, q_len, q_heads * dim)


def test_attention_matches_reference():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((2, 3, 4, 8)).astype(np.float32)
    k = rng.standard_normal((2, 2, 5, 8)).astype(np.float32)
    v = rng.standard_normal((2, 2, 5, 8)).astype(np.float32)
    mask = np.asarray(causal_mask(3, 5, past_len=2))
    expected = _reference_attention(q, k, v, mask, 2, 8 ** -0.5)
    actual = np.asarray(attention(
        jnp.asarray(q), jnp.asarray(k), jnp.asarray(v),
        jnp.asarray(mask), 2, 8 ** -0.5))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_swiglu_matches_reference():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 4)).astype(np.float32)
    gw = rng.standard_normal((4, 6)).astype(np.float32)
    uw = rng.standard_normal((4, 6)).astype(np.float32)
    dw = rng.standard_normal((6, 4)).astype(np.float32)
    gate = x @ gw
    up = x @ uw
    expected = (gate * (1 / (1 + np.exp(-gate))) * up) @ dw
    actual = np.asarray(swiglu(jnp.asarray(x), jnp.asarray(gw),
                              jnp.asarray(uw), jnp.asarray(dw)))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_silu_matches_reference():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    expected = x / (1 + np.exp(-x))
    np.testing.assert_allclose(np.asarray(silu(jnp.asarray(x))), expected,
                               rtol=1e-6, atol=1e-6)
