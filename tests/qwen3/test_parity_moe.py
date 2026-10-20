# -*- coding: utf-8 -*-
"""Numeric parity between the JAX MoE model and HuggingFace transformers."""
import numpy as np
import pytest

from helpers import build_hf_moe, export_hf_checkpoint
from sunstreaker.applications.transformers.qwen3 import Qwen3MoeForCausalLM

IDS = np.array([[1, 5, 9, 17, 3, 42, 7, 11]], dtype=np.int32)


@pytest.fixture(scope="module")
def moe_pair(tmp_path_factory):
    directory = tmp_path_factory.mktemp("moe")
    hf_model = build_hf_moe()
    export_hf_checkpoint(hf_model, str(directory))
    model = Qwen3MoeForCausalLM.from_pretrained(str(directory), dtype="float32")
    return hf_model, model


def _hf_forward(model, ids, attention_mask=None):
    import torch

    inputs = torch.tensor(ids, dtype=torch.long)
    mask = None
    if attention_mask is not None:
        mask = torch.tensor(attention_mask, dtype=torch.long)
    model.config.output_router_logits = True
    with torch.no_grad():
        out = model(input_ids=inputs, attention_mask=mask,
                    output_router_logits=True)
    return out.logits.float().numpy(), float(out.aux_loss)


def test_expert_weights_are_stacked(moe_pair):
    _, model = moe_pair
    assert model.params["model.layers.0.mlp.experts.gate_proj.weight"].shape == (
        4, 64, 32)
    assert model.params["model.layers.0.mlp.experts.down_proj.weight"].shape == (
        4, 32, 64)
    assert model.params["model.layers.0.mlp.gate.weight"].shape == (4, 64)


def test_logits_and_aux_loss_match_hf(moe_pair):
    hf_model, model = moe_pair
    ref_logits, ref_aux = _hf_forward(hf_model, IDS)
    out = model(IDS, use_cache=False, output_router_logits=True)
    np.testing.assert_allclose(np.asarray(out["logits"]), ref_logits, atol=1e-5)
    np.testing.assert_allclose(float(out["aux_loss"]), ref_aux, rtol=1e-6)


def test_masked_aux_loss_matches_hf(moe_pair):
    hf_model, model = moe_pair
    mask = np.array([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=np.int32)
    ref_logits, ref_aux = _hf_forward(hf_model, IDS, mask)
    out = model(IDS, attention_mask=mask, use_cache=False,
                output_router_logits=True)
    np.testing.assert_allclose(np.asarray(out["logits"]), ref_logits, atol=1e-5)
    np.testing.assert_allclose(float(out["aux_loss"]), ref_aux, rtol=1e-6)


def test_cached_forward_matches_full(moe_pair):
    _, model = moe_pair
    full = np.asarray(model(IDS, use_cache=False)["logits"])

    first = model(IDS[:, :4], use_cache=True)
    second = model(
        IDS[:, 4:],
        attention_mask=np.ones((1, IDS.shape[1]), dtype=np.int32),
        position_ids=np.arange(4, IDS.shape[1], dtype=np.int32),
        past_key_values=first["past_key_values"],
        use_cache=True,
    )
    cached = np.concatenate(
        [np.asarray(first["logits"]), np.asarray(second["logits"])], axis=1)
    np.testing.assert_allclose(cached, full, atol=1e-5)


def test_greedy_generation_matches_hf(moe_pair):
    import torch

    hf_model, model = moe_pair
    with torch.no_grad():
        ref = hf_model.generate(
            torch.tensor(IDS, dtype=torch.long), max_new_tokens=6,
            do_sample=False, eos_token_id=None,
            pad_token_id=hf_model.config.pad_token_id or 0,
        )[0].numpy()
    ours = np.asarray(model.generate(
        IDS, max_new_tokens=6, temperature=0.0, eos_token_id=None))
    np.testing.assert_array_equal(ours[0], ref)
