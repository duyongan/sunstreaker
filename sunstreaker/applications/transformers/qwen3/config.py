# -*- coding: utf-8 -*-
"""Qwen3 / Qwen3-MoE configuration."""
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

CONFIG_NAME = "config.json"


@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 40960
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_theta: float = 1e6
    rope_scaling: Optional[dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: Optional[int] = None
    use_sliding_window: bool = False
    max_window_layers: Optional[int] = None
    model_type: str = "qwen3"
    torch_dtype: str = "bfloat16"

    # MoE fields
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = False
    decoder_sparse_step: int = 1
    mlp_only_layers: Tuple[int, ...] = field(default_factory=tuple)
    router_aux_loss_coef: float = 0.001
    output_router_logits: bool = False

    # Special tokens
    bos_token_id: Optional[int] = 151643
    eos_token_id: Optional[int] = 151645
    pad_token_id: Optional[int] = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if isinstance(self.mlp_only_layers, list):
            self.mlp_only_layers = tuple(self.mlp_only_layers)
        if self.model_type == "qwen3_moe" and self.num_experts <= 0:
            raise ValueError("qwen3_moe 配置缺少 num_experts")

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def layer_is_moe(self, layer_idx: int) -> bool:
        if not self.is_moe:
            return False
        if layer_idx in self.mlp_only_layers:
            return False
        return (layer_idx + 1) % self.decoder_sparse_step == 0

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Qwen3Config":
        config_dict = dict(config_dict)
        # transformers>=4.55 nests RoPE settings, and 5.x renamed
        # ``torch_dtype`` to ``dtype``.
        rope_params = config_dict.get("rope_parameters")
        if isinstance(rope_params, dict):
            config_dict.setdefault(
                "rope_theta", rope_params.get("rope_theta", 1e6))
        if "torch_dtype" not in config_dict and isinstance(config_dict.get("dtype"), str):
            config_dict["torch_dtype"] = config_dict["dtype"]
        # transformers 5.x renamed MoE config's ``num_experts``.
        if "num_experts" not in config_dict and "num_local_experts" in config_dict:
            config_dict["num_experts"] = config_dict["num_local_experts"]
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in config_dict.items() if k in known})

    @classmethod
    def from_json_file(cls, path: str) -> "Qwen3Config":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str,
        *,
        endpoint: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> "Qwen3Config":
        from .loading import resolve_model_dir

        model_dir = resolve_model_dir(
            path_or_repo_id, endpoint=endpoint, revision=revision, cache_dir=cache_dir
        )
        return cls.from_json_file(os.path.join(model_dir, CONFIG_NAME))

    def to_dict(self) -> dict:
        config_dict = {
            k: v for k, v in asdict(self).items() if v is not None
        }
        # Keep exports loadable by both transformers 4.x and 5.x.
        config_dict.setdefault("dtype", self.torch_dtype)
        config_dict.setdefault(
            "rope_parameters",
            {"rope_theta": self.rope_theta, "rope_type": "default"})
        if self.is_moe:
            config_dict["num_local_experts"] = self.num_experts
        return config_dict

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, CONFIG_NAME), "w", encoding="utf-8") as f:
            f.write(self.to_json_string())
