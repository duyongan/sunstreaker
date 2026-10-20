# -*- coding: utf-8 -*-
"""Numeric parity between the JAX dense model and HuggingFace transformers."""
import numpy as np
import pytest

from helpers import build_hf_dense, export_hf_checkpoint
from sunstreaker.applications.transformers.qwen3 import Qwen3ForCausalLM

IDS = np.array([[1, 5, 9, 17, 3, 42]], dtype=np.int32)


@pytest.fixture(scope="module")
def dense_pair(tmp_path_factory):
    directory = tmp_path_factory.mktemp("dense")
    hf_model = build_hf_dense()
    export_hf_checkpoint(hf_model, str(directory))
    model = Qwen3ForCausalLM.from_pretrained(str(directory), dtype="float32")
    return hf_model, model


def _hf_logits(model, ids, attention_mask=None):
    import torch

    inputs = torch.tensor(ids, dtype=torch.long)
    mask = None
    if attention_mask is not None:
        mask = torch.tensor(attention_mask, dtype=torch.long)
    model.config.output_router_logits = False
    with torch.no_grad():
        out = model(input_ids=inputs, attention_mask=mask)
    return out.logits.float().numpy()


def test_logits_match_hf(dense_pair):
    hf_model, model = dense_pair
    ref = _hf_logits(hf_model, IDS)
    out = model(IDS, use_cache=False)
    np.testing.assert_allclose(np.asarray(out["logits"]), ref, atol=1e-5)


def test_masked_logits_match_hf(dense_pair):
    hf_model, model = dense_pair
    mask = np.array([[1, 1, 1, 0, 0, 0]], dtype=np.int32)
    ref = _hf_logits(hf_model, IDS, mask)
    out = model(IDS, attention_mask=mask, use_cache=False)
    np.testing.assert_allclose(np.asarray(out["logits"]), ref, atol=1e-5)


def test_kv_cache_matches_full_forward(dense_pair):
    _, model = dense_pair
    full = np.asarray(model(IDS, use_cache=False)["logits"])

    first = model(IDS[:, :3], use_cache=True)
    second = model(
        IDS[:, 3:],
        attention_mask=np.ones((1, IDS.shape[1]), dtype=np.int32),
        position_ids=np.arange(3, IDS.shape[1], dtype=np.int32),
        past_key_values=first["past_key_values"],
        use_cache=True,
    )
    cached = np.concatenate(
        [np.asarray(first["logits"]), np.asarray(second["logits"])], axis=1)
    np.testing.assert_allclose(cached, full, atol=1e-5)


def test_greedy_generation_matches_hf(dense_pair):
    import torch

    hf_model, model = dense_pair
    with torch.no_grad():
        ref = hf_model.generate(
            torch.tensor(IDS, dtype=torch.long), max_new_tokens=6,
            do_sample=False, eos_token_id=None,
            pad_token_id=hf_model.config.pad_token_id or 0,
        )[0].numpy()
    ours = np.asarray(model.generate(
        IDS, max_new_tokens=6, temperature=0.0, eos_token_id=None))
    np.testing.assert_array_equal(ours[0], ref)


def test_tied_embeddings(dense_pair):
    _, model = dense_pair
    embed = np.asarray(model.params["model.embed_tokens.weight"])
    head = np.asarray(model.params["lm_head.weight"])
    np.testing.assert_array_equal(head, embed.T)
