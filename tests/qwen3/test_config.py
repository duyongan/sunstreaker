# -*- coding: utf-8 -*-
import json

import pytest

from sunstreaker.applications.transformers.qwen3 import Qwen3Config


def test_defaults_and_properties():
    config = Qwen3Config(hidden_size=64, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16)
    assert config.is_moe is False
    assert config.num_key_value_groups == 2
    assert config.layer_is_moe(0) is False


def test_from_dict_handles_transformers_5_fields():
    config = Qwen3Config.from_dict({
        "vocab_size": 128,
        "hidden_size": 64,
        "model_type": "qwen3_moe",
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
        "dtype": "bfloat16",
    })
    assert config.num_experts == 4
    assert config.rope_theta == 500000.0
    assert config.torch_dtype == "bfloat16"
    assert config.is_moe


def test_layer_is_moe_respects_sparse_step():
    config = Qwen3Config(num_experts=8, num_experts_per_tok=2,
                         moe_intermediate_size=32, decoder_sparse_step=2,
                         mlp_only_layers=(3,))
    assert config.layer_is_moe(0) is False
    assert config.layer_is_moe(1) is True
    assert config.layer_is_moe(3) is False
    assert config.layer_is_moe(5) is True


def test_json_roundtrip(tmp_path):
    config = Qwen3Config(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                         num_attention_heads=2, head_dim=8, num_experts=4)
    config.save_pretrained(str(tmp_path))
    raw = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert "rope_parameters" in raw
    assert raw["rope_parameters"]["rope_theta"] == config.rope_theta
    assert raw["num_local_experts"] == 4
    assert "max_window_layers" not in raw  # None fields are omitted

    loaded = Qwen3Config.from_json_file(str(tmp_path / "config.json"))
    assert loaded.num_experts == 4
    assert loaded.hidden_size == 16


def test_moe_config_requires_experts():
    with pytest.raises(ValueError):
        Qwen3Config(model_type="qwen3_moe")
