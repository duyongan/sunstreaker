# -*- coding: utf-8 -*-
"""Offline tests for causality, KV cache and generation helpers."""
import numpy as np

from helpers import DENSE_KWARGS
from sunstreaker.applications.transformers.qwen3 import (
    Qwen3Config,
    Qwen3ForCausalLM,
)

IDS = np.array([[1, 5, 9, 17, 3, 42]], dtype=np.int32)


def _model():
    config = Qwen3Config(**DENSE_KWARGS)
    return Qwen3ForCausalLM(config)


def test_causality():
    model = _model()
    base = np.asarray(model(IDS, use_cache=False)["logits"])
    changed = IDS.copy()
    changed[0, 3:] = (changed[0, 3:] + 7) % model.config.vocab_size
    other = np.asarray(model(changed, use_cache=False)["logits"])
    np.testing.assert_allclose(base[:, :3], other[:, :3], atol=1e-5)
    assert not np.allclose(base[:, 3:], other[:, 3:])


def test_kv_cache_matches_full_forward():
    model = _model()
    full = np.asarray(model(IDS, use_cache=False)["logits"])

    first = model(IDS[:, :2], use_cache=True)
    second = model(
        IDS[:, 2:],
        attention_mask=np.ones((1, IDS.shape[1]), dtype=np.int32),
        position_ids=np.arange(2, IDS.shape[1], dtype=np.int32),
        past_key_values=first["past_key_values"],
        use_cache=True,
    )
    cached = np.concatenate(
        [np.asarray(first["logits"]), np.asarray(second["logits"])], axis=1)
    np.testing.assert_allclose(cached, full, atol=1e-5)


def test_greedy_is_deterministic():
    model = _model()
    first = np.asarray(model.generate(
        IDS, max_new_tokens=5, temperature=0.0, eos_token_id=None))
    second = np.asarray(model.generate(
        IDS, max_new_tokens=5, temperature=0.0, eos_token_id=None))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (1, IDS.shape[1] + 5)


def test_cache_and_no_cache_generation_agree():
    model = _model()
    with_cache = np.asarray(model.generate(
        IDS, max_new_tokens=4, temperature=0.0, eos_token_id=None,
        use_cache=True))
    without_cache = np.asarray(model.generate(
        IDS, max_new_tokens=4, temperature=0.0, eos_token_id=None,
        use_cache=False))
    np.testing.assert_array_equal(with_cache, without_cache)


def test_eos_stops_generation():
    model = _model()
    logits = model(IDS, use_cache=False)["logits"]
    eos = int(np.argmax(np.asarray(logits[0, -1])))
    out = model.generate(IDS, max_new_tokens=8, temperature=0.0,
                         eos_token_id=eos)
    assert out.shape == (1, IDS.shape[1] + 1)
    assert int(out[0, -1]) == eos


def test_top_k_and_top_p_sampling():
    model = _model()
    ids = model.generate(IDS, max_new_tokens=4, temperature=0.8,
                         top_k=5, top_p=0.9)
    assert ids.shape == (1, IDS.shape[1] + 4)
    assert np.all((np.asarray(ids) >= 0)
                  & (np.asarray(ids) < model.config.vocab_size))
