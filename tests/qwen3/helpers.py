# -*- coding: utf-8 -*-
"""Shared helpers for Qwen3 tests."""
import os

import pytest

DENSE_KWARGS = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=64,
    tie_word_embeddings=True,
    rope_theta=10000.0,
    rms_norm_eps=1e-6,
    attention_bias=False,
    torch_dtype="float32",
)

MOE_KWARGS = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=64,
    tie_word_embeddings=False,
    rope_theta=10000.0,
    rms_norm_eps=1e-6,
    attention_bias=False,
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    norm_topk_prob=True,
    decoder_sparse_step=1,
    mlp_only_layers=[],
    torch_dtype="float32",
)


def import_hf():
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    return transformers, torch


def build_hf_dense(seed=0):
    transformers, torch = import_hf()
    config = transformers.Qwen3Config(attn_implementation="eager", **DENSE_KWARGS)
    torch.manual_seed(seed)
    return transformers.Qwen3ForCausalLM(config).eval()


def build_hf_moe(seed=0):
    transformers, torch = import_hf()
    config = transformers.Qwen3MoeConfig(
        attn_implementation="eager", **MOE_KWARGS)
    torch.manual_seed(seed)
    return transformers.Qwen3MoeForCausalLM(config).eval()


def export_hf_checkpoint(model, directory):
    """Write a HF state dict + config to ``directory`` (legacy layout)."""
    from safetensors.torch import save_file

    os.makedirs(directory, exist_ok=True)
    tied = getattr(model.config, "tie_word_embeddings", False)
    state = {}
    for name, tensor in model.state_dict().items():
        if tied and name == "lm_head.weight":
            continue
        state[name] = tensor.detach().contiguous()
    save_file(state, os.path.join(directory, "model.safetensors"))
    model.config.save_pretrained(directory)
