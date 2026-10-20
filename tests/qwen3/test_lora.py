# -*- coding: utf-8 -*-
"""Tests for LoRA adapters on dense and MoE Qwen3 models."""
import jax
import jax.numpy as jnp
import numpy as np

from helpers import DENSE_KWARGS, MOE_KWARGS
from sunstreaker.applications.transformers.qwen3 import (
    LoRAConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeForCausalLM,
)

IDS = jnp.asarray([[1, 5, 9, 17, 3, 42]])


def _dense_model(**overrides):
    config = Qwen3Config(**{**DENSE_KWARGS, **overrides})
    return Qwen3ForCausalLM(config)


def _moe_model(**overrides):
    config = Qwen3Config(**{**MOE_KWARGS, **overrides})
    return Qwen3MoeForCausalLM(config)


def test_lora_config_defaults():
    config = LoRAConfig(rank=4, alpha=8.0)
    assert config.scale == 2.0
    assert "q_proj" in config.target_modules


def test_zero_init_matches_base_logits():
    model = _dense_model()
    base = np.asarray(model(IDS, use_cache=False)["logits"])
    model.add_lora(rank=4, alpha=8.0)
    adapted = np.asarray(model(IDS, use_cache=False)["logits"])
    np.testing.assert_array_equal(adapted, base)


def test_target_modules_filter():
    model = _dense_model()
    model.add_lora(rank=2, target_modules=("q_proj",))
    keys = set(model.lora_params)
    assert len(keys) == model.config.num_hidden_layers
    assert all(name.endswith("self_attn.q_proj.weight") for name in keys)
    pair = model.lora_params["model.layers.0.self_attn.q_proj.weight"]
    hidden = model.config.hidden_size
    assert pair["A"].shape == (hidden, 2)
    assert pair["B"].shape == (2, hidden)


def test_fit_trains_only_adapters():
    model = _dense_model(num_hidden_layers=1)
    model.add_lora(rank=4, alpha=8.0)
    base_before = {k: np.array(v) for k, v in model.params.items()}
    adapters_before = jax.tree.map(lambda x: np.array(x), model.lora_params)
    batch = (IDS, jnp.ones(IDS.shape, dtype=jnp.int32))

    history = model.fit([batch], epochs=12, lr=0.1, weight_decay=0.0,
                        clip_norm=None)

    assert history[-1]["loss"] < history[0]["loss"]
    for name, value in model.params.items():
        np.testing.assert_array_equal(np.asarray(value), base_before[name])
    changed = [
        name for name in adapters_before
        if not np.array_equal(np.asarray(model.lora_params[name]["B"]),
                              adapters_before[name]["B"])
    ]
    assert changed


def test_merge_matches_adapted_forward():
    model = _dense_model(num_hidden_layers=1)
    model.add_lora(rank=4, alpha=8.0)
    batch = (IDS, jnp.ones(IDS.shape, dtype=jnp.int32))
    model.fit([batch], epochs=5, lr=0.1, weight_decay=0.0, clip_norm=None)

    adapted = np.asarray(model(IDS, use_cache=False)["logits"])
    model.merge_lora()
    assert model.lora_params is None
    assert model.lora_config is None
    merged = np.asarray(model(IDS, use_cache=False)["logits"])
    np.testing.assert_allclose(merged, adapted, rtol=1e-4, atol=1e-4)


def test_moe_lora_shapes_and_training():
    model = _moe_model(num_hidden_layers=1)
    base = np.asarray(model(IDS, use_cache=False)["logits"])
    model.add_lora(rank=4, alpha=8.0)

    hidden = model.config.hidden_size
    inter = model.config.moe_intermediate_size
    experts = model.config.num_experts
    gate = model.lora_params["model.layers.0.mlp.experts.gate_proj.weight"]
    down = model.lora_params["model.layers.0.mlp.experts.down_proj.weight"]
    assert gate["A"].shape == (experts, hidden, 4)
    assert gate["B"].shape == (experts, 4, inter)
    assert down["A"].shape == (experts, inter, 4)
    assert down["B"].shape == (experts, 4, hidden)

    adapted = np.asarray(model(IDS, use_cache=False)["logits"])
    np.testing.assert_array_equal(adapted, base)

    batch = (IDS, jnp.ones(IDS.shape, dtype=jnp.int32))
    history = model.fit([batch], epochs=2, lr=0.05, weight_decay=0.0,
                        clip_norm=None)
    assert np.isfinite(history[-1]["loss"])
    assert any(float(jnp.sum(jnp.abs(pair["B"]))) > 0.0
               for pair in model.lora_params.values())


def test_merge_then_save_roundtrip(tmp_path):
    model = _dense_model(num_hidden_layers=1)
    model.add_lora(rank=2, alpha=4.0)
    batch = (IDS, jnp.ones(IDS.shape, dtype=jnp.int32))
    model.fit([batch], epochs=3, lr=0.1, weight_decay=0.0, clip_norm=None)
    model.merge_lora()

    expected = np.asarray(model(IDS, use_cache=False)["logits"])
    model.save_pretrained(str(tmp_path), dtype="float32")
    reloaded = Qwen3ForCausalLM.from_pretrained(str(tmp_path), dtype="float32")
    actual = np.asarray(reloaded(IDS, use_cache=False)["logits"])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
