# -*- coding: utf-8 -*-
"""Tokenizer and chat-template helpers for Qwen3.

Qwen3 uses a byte-level BPE tokenizer, so instead of reimplementing it we
delegate to the optional ``transformers`` dependency.
"""
from typing import Any, List, Optional

from .loading import resolve_model_dir


class AutoTokenizer:
    """Thin wrapper around ``transformers.AutoTokenizer``."""

    @staticmethod
    def from_pretrained(path_or_repo_id: str, *, endpoint: Optional[str] = None,
                        revision: Optional[str] = None,
                        cache_dir: Optional[str] = None, **kwargs):
        try:
            from transformers import AutoTokenizer as _HFAutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "使用 Qwen3 tokenizer 需要安装 transformers："
                "pip install 'transformers>=4.51'") from exc

        model_dir = resolve_model_dir(
            path_or_repo_id, endpoint=endpoint, revision=revision,
            cache_dir=cache_dir)
        kwargs.setdefault("use_fast", True)
        kwargs.setdefault("trust_remote_code", False)
        return _HFAutoTokenizer.from_pretrained(model_dir, **kwargs)


def apply_chat_template(tokenizer, messages: List[dict],
                        add_generation_prompt: bool = True,
                        tokenize: bool = False, **kwargs: Any):
    """Render chat messages with the tokenizer's built-in Qwen chat template."""
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError("tokenizer 不支持 apply_chat_template")
    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
