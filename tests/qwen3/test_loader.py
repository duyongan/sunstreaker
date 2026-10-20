# -*- coding: utf-8 -*-
"""Tests for safetensors loading, sharding and export."""
import json
import os

import numpy as np
import pytest

from helpers import build_hf_dense, build_hf_moe, export_hf_checkpoint
from sunstreaker.applications.transformers.qwen3 import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeForCausalLM,
)

IDS = np.array([[1, 5, 9, 17, 3]], dtype=np.int32)


def _torch_state(path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    return load_file(os.path.join(path, "model.safetensors"))


def _save_torch_state(state, path):
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in state.items()},
              os.path.join(path, "model.safetensors"))


@pytest.fixture(scope="module")
def dense_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("dense-export")
    hf_model = build_hf_dense()
    export_hf_checkpoint(hf_model, str(directory))
    return hf_model, directory


def test_linear_weights_are_transposed(dense_dir):
    hf_model, directory = dense_dir
    model = Qwen3ForCausalLM.from_pretrained(str(directory), dtype="float32")
    ours = np.asarray(model.params["model.layers.0.self_attn.q_proj.weight"])
    ref = hf_model.model.layers[0].self_attn.q_proj.weight.detach().numpy().T
    np.testing.assert_allclose(ours, ref, atol=0)
    np.testing.assert_array_equal(
        np.asarray(model.params["model.embed_tokens.weight"]),
        hf_model.model.embed_tokens.weight.detach().numpy())


def test_rotary_inv_freq_is_ignored(dense_dir, tmp_path):
    _, directory = dense_dir
    state = _torch_state(directory)
    state = dict(state)
    state["model.layers.0.self_attn.rotary_emb.inv_freq"] = state[
        "model.norm.weight"].clone()
    (tmp_path / "config.json").write_text(
        (directory / "config.json").read_text(encoding="utf-8"), encoding="utf-8")
    _save_torch_state(state, tmp_path)
    model = Qwen3ForCausalLM.from_pretrained(str(tmp_path), dtype="float32")
    assert "model.layers.0.self_attn.rotary_emb.inv_freq" not in model.params


def test_missing_weights_raise(dense_dir, tmp_path):
    _, directory = dense_dir
    state = dict(_torch_state(directory))
    state.pop("model.norm.weight")
    cfg = (directory / "config.json").read_text(encoding="utf-8")
    (tmp_path / "config.json").write_text(cfg, encoding="utf-8")
    _save_torch_state(state, tmp_path)
    with pytest.raises(ValueError, match="权重缺失"):
        Qwen3ForCausalLM.from_pretrained(str(tmp_path), dtype="float32")


def test_sharded_index_is_supported(dense_dir, tmp_path):
    _, directory = dense_dir
    state = _torch_state(directory)
    names = list(state)
    half = len(names) // 2
    shards = {
        "model-00001-of-00002.safetensors": names[:half],
        "model-00002-of-00002.safetensors": names[half:],
    }
    weight_map = {}
    from safetensors.torch import save_file

    for shard, shard_names in shards.items():
        for name in shard_names:
            weight_map[name] = shard
    for shard, shard_names in shards.items():
        save_file({n: state[n].contiguous() for n in shard_names},
                  os.path.join(tmp_path, shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
    (tmp_path / "config.json").write_text(
        (directory / "config.json").read_text(encoding="utf-8"), encoding="utf-8")

    single = Qwen3ForCausalLM.from_pretrained(str(directory), dtype="float32")
    sharded = Qwen3ForCausalLM.from_pretrained(str(tmp_path), dtype="float32")
    np.testing.assert_allclose(
        np.asarray(single(IDS, use_cache=False)["logits"]),
        np.asarray(sharded(IDS, use_cache=False)["logits"]), atol=0)


def test_save_pretrained_roundtrip(dense_dir, tmp_path):
    _, directory = dense_dir
    model = Qwen3ForCausalLM.from_pretrained(str(directory), dtype="float32")
    model.save_pretrained(str(tmp_path), dtype="float32")

    raw = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert all(value is not None for value in raw.values())
    assert raw["rope_parameters"]["rope_theta"] == 10000.0

    saved = _torch_state(tmp_path)
    assert "lm_head.weight" not in saved  # tied weights are omitted
    np.testing.assert_allclose(
        saved["model.layers.0.self_attn.q_proj.weight"].numpy(),
        np.asarray(model.params["model.layers.0.self_attn.q_proj.weight"]).T,
        atol=0)

    reloaded = Qwen3ForCausalLM.from_pretrained(str(tmp_path), dtype="float32")
    np.testing.assert_allclose(
        np.asarray(model(IDS, use_cache=False)["logits"]),
        np.asarray(reloaded(IDS, use_cache=False)["logits"]), atol=0)


def test_moe_export_and_reload(tmp_path):
    hf_model = build_hf_moe()
    export_hf_checkpoint(hf_model, str(tmp_path / "hf"))
    model = Qwen3MoeForCausalLM.from_pretrained(
        str(tmp_path / "hf"), dtype="float32")
    assert model.config.is_moe

    out_dir = tmp_path / "ours"
    model.save_pretrained(str(out_dir), dtype="float32")
    from safetensors.numpy import load_file

    saved = load_file(str(out_dir / "model.safetensors"))
    stacked = np.asarray(
        model.params["model.layers.0.mlp.experts.gate_proj.weight"])
    for idx in range(stacked.shape[0]):
        np.testing.assert_allclose(
            saved[f"model.layers.0.mlp.experts.{idx}.gate_proj.weight"],
            stacked[idx].T, atol=0)

    reloaded = Qwen3MoeForCausalLM.from_pretrained(
        str(out_dir), dtype="float32")
    ids = np.array([[1, 5, 9, 17, 3]], dtype=np.int32)
    np.testing.assert_allclose(
        np.asarray(model(ids, use_cache=False)["logits"]),
        np.asarray(reloaded(ids, use_cache=False)["logits"]), atol=0)


def test_config_roundtrip_drops_none(tmp_path):
    config = Qwen3Config(vocab_size=32, hidden_size=16, num_hidden_layers=1)
    config.save_pretrained(str(tmp_path))
    loaded = Qwen3Config.from_json_file(str(tmp_path / "config.json"))
    assert loaded.max_window_layers is None
    assert loaded.vocab_size == 32
    assert loaded.model_type == "qwen3"
