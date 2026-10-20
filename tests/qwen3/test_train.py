# -*- coding: utf-8 -*-
"""Offline tests for the loss and the optax training loop."""
import jax
import jax.numpy as jnp
import numpy as np

from helpers import DENSE_KWARGS, MOE_KWARGS
from sunstreaker.applications.transformers.qwen3 import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeForCausalLM,
)
from sunstreaker.applications.transformers.qwen3.train import (
    IGNORE_INDEX,
    causal_lm_loss,
    pad_batch,
)

IDS = jnp.asarray([[1, 5, 9, 17, 3, 42]])


def test_pad_batch():
    ids, mask = pad_batch([[1, 2, 3], [4, 5]], pad_token_id=0)
    np.testing.assert_array_equal(np.asarray(ids), [[1, 2, 3], [4, 5, 0]])
    np.testing.assert_array_equal(np.asarray(mask), [[1, 1, 1], [1, 1, 0]])

    _, mask = pad_batch([[1, 2, 3]], max_length=5)
    np.testing.assert_array_equal(np.asarray(mask), [[1, 1, 1, 0, 0]])


def test_loss_and_gradients_are_finite():
    config = Qwen3Config(**DENSE_KWARGS)
    model = Qwen3ForCausalLM(config)

    loss = causal_lm_loss(config, model.params, IDS)
    assert np.isfinite(float(loss))

    grads = jax.grad(
        lambda p: causal_lm_loss(config, p, IDS))(model.params)
    total = sum(float(jnp.sum(jnp.abs(g))) for g in jax.tree.leaves(grads))
    assert total > 0.0
    assert all(np.all(np.isfinite(np.asarray(g)))
               for g in jax.tree.leaves(grads))


def test_padding_and_ignore_labels_do_not_contribute():
    config = Qwen3Config(**DENSE_KWARGS)
    model = Qwen3ForCausalLM(config)
    labels = jnp.full(IDS.shape, IGNORE_INDEX, dtype=jnp.int32)
    loss = causal_lm_loss(config, model.params, IDS, labels=labels)
    assert float(loss) == 0.0

    mask = jnp.asarray([[1, 1, 1, 1, 0, 0]], dtype=jnp.int32)
    masked = causal_lm_loss(config, model.params, IDS, attention_mask=mask)
    assert np.isfinite(float(masked))


def test_fit_reduces_loss():
    config = Qwen3Config(**{**DENSE_KWARGS, "num_hidden_layers": 1})
    model = Qwen3ForCausalLM(config)
    batch = (IDS, jnp.ones(IDS.shape, dtype=jnp.int32))
    history = model.fit([batch], epochs=8, lr=0.05, weight_decay=0.0,
                        clip_norm=None)
    assert len(history) == 8
    assert history[-1]["loss"] < history[0]["loss"]


def test_moe_loss_includes_aux_loss():
    config = Qwen3Config(**{**MOE_KWARGS, "output_router_logits": True})
    model = Qwen3MoeForCausalLM(config)
    loss = causal_lm_loss(config, model.params, IDS)
    assert np.isfinite(float(loss))

    out = model(IDS, use_cache=False, output_router_logits=True)
    assert float(out["aux_loss"]) > 0.0
    assert float(out["aux_loss"]) < 10.0
