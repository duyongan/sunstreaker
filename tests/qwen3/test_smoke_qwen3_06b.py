# -*- coding: utf-8 -*-
"""End-to-end smoke test against the real Qwen3-0.6B checkpoint.

Network test: enable with ``RUN_QWEN3_SMOKE=1``.  Downloads ~1.5 GB from
``HF_ENDPOINT`` (hf-mirror by default).
"""
import os

import numpy as np
import pytest

from sunstreaker.applications.transformers.qwen3 import (
    AutoTokenizer,
    Qwen3ForCausalLM,
    apply_chat_template,
)

RUN_SMOKE = os.environ.get("RUN_QWEN3_SMOKE") == "1"
REPO_ID = os.environ.get("QWEN3_SMOKE_REPO", "Qwen/Qwen3-0.6B")

pytestmark = pytest.mark.skipif(
    not RUN_SMOKE, reason="设置 RUN_QWEN3_SMOKE=1 以运行联网 smoke 测试")


@pytest.fixture(scope="module")
def loaded():
    model = Qwen3ForCausalLM.from_pretrained(REPO_ID, dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    return model, tokenizer


def test_weights_loaded(loaded):
    model, _ = loaded
    assert model.config.model_type == "qwen3"
    assert model.config.num_hidden_layers > 0
    assert str(model.params["model.embed_tokens.weight"].dtype) == "bfloat16"


def test_chat_template_and_greedy_generation(loaded):
    model, tokenizer = loaded
    prompt = apply_chat_template(
        tokenizer, [{"role": "user", "content": "用一句话介绍你自己。"}])
    ids = tokenizer(prompt, return_tensors="np")["input_ids"].astype(np.int32)
    out = model.generate(ids, max_new_tokens=8, temperature=0.0,
                         eos_token_id=None)
    text = tokenizer.decode(np.asarray(out[0], dtype=np.int32))
    assert text.startswith(prompt)
    assert len(text) > len(prompt)


def test_first_token_matches_transformers(loaded):
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    model, tokenizer = loaded
    ids = tokenizer("Hello", return_tensors="np")["input_ids"].astype(np.int32)

    ours = np.asarray(model(ids, use_cache=False)["logits"][0, -1])

    hf_model = transformers.Qwen3ForCausalLM.from_pretrained(
        REPO_ID, dtype=torch.bfloat16)
    hf_model.eval()
    with torch.no_grad():
        ref = hf_model(input_ids=torch.tensor(ids, dtype=torch.long))
    ref = ref.logits[0, -1].float().numpy()

    assert int(np.argmax(ours)) == int(np.argmax(ref))
    np.testing.assert_allclose(ours, ref, atol=0.5)
