# -*- coding: utf-8 -*-
"""LoRA adapters for the Qwen3 linear projections.

Adapters live in their own pytree keyed by the base parameter name, e.g.
``model.layers.0.self_attn.q_proj.weight -> {"A": (in, r), "B": (r, out)}``.
MoE expert projections are stacked over experts, so their adapters have an
extra leading axis ``(E, in, r)`` / ``(E, r, out)``.

``B`` starts at zero so the adapted model is numerically identical to the
base model before training.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp

DEFAULT_TARGET_MODULES: Tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

LoRAParams = Dict[str, Dict[str, jnp.ndarray]]


@dataclass
class LoRAConfig:
    """Hyper-parameters for the low-rank adapters."""

    rank: int = 8
    alpha: float = 16.0
    target_modules: Tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_TARGET_MODULES)
    init_std: float = 0.01

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def __post_init__(self):
        if self.rank <= 0:
            raise ValueError("rank 必须为正整数")
        if self.alpha <= 0:
            raise ValueError("alpha 必须为正数")
        self.target_modules = tuple(self.target_modules)


def _module_name(param_name: str) -> str:
    return param_name.split(".")[-2]


def init_lora_params(
    params: Dict[str, jnp.ndarray],
    config: Optional[LoRAConfig] = None,
    *,
    rng: Optional[jnp.ndarray] = None,
) -> LoRAParams:
    """Create zero-output adapters for every targeted linear weight."""
    config = config or LoRAConfig()
    if rng is None:
        rng = jax.random.PRNGKey(0)

    lora: LoRAParams = {}
    for name, weight in params.items():
        if not name.endswith(".weight"):
            continue
        if _module_name(name) not in config.target_modules:
            continue
        if weight.ndim == 2:
            in_dim, out_dim = weight.shape
            prefix = ()
        elif weight.ndim == 3:
            n_exp, in_dim, out_dim = weight.shape
            prefix = (n_exp,)
        else:
            raise ValueError(f"不支持的权重形状 {name}: {weight.shape}")

        rng, key_a = jax.random.split(rng)
        a = (jax.random.normal(key_a, prefix + (in_dim, config.rank),
                               dtype=jnp.float32) * config.init_std)
        lora[name] = {
            "A": a.astype(weight.dtype),
            "B": jnp.zeros(prefix + (config.rank, out_dim), dtype=weight.dtype),
        }

    if not lora:
        raise ValueError(
            "没有匹配到任何目标模块，请检查 target_modules 或参数名")
    return lora


def merge_lora_params(
    params: Dict[str, jnp.ndarray],
    lora_params: LoRAParams,
    scale: float,
) -> Dict[str, jnp.ndarray]:
    """Return a copy of ``params`` with the adapter deltas merged in."""
    merged = dict(params)
    for name, pair in lora_params.items():
        if name not in merged:
            raise ValueError(f"适配器对应的基础权重不存在: {name}")
        weight = merged[name]
        a, b = pair["A"], pair["B"]
        if weight.ndim == 2:
            delta = jnp.einsum("ir,ro->io", a, b, preferred_element_type=jnp.float32)
        else:
            delta = jnp.einsum("eir,ero->eio", a, b,
                               preferred_element_type=jnp.float32)
        merged[name] = weight + (scale * delta).astype(weight.dtype)
    return merged
