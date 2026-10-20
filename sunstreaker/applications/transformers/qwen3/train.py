# -*- coding: utf-8 -*-
"""Causal-LM loss and a small optax training loop for Qwen3."""
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax

from .config import Qwen3Config
from .modeling import forward

IGNORE_INDEX = -100


def causal_lm_loss(
    config: Qwen3Config,
    params: Dict[str, jnp.ndarray],
    input_ids: jnp.ndarray,
    attention_mask: Optional[jnp.ndarray] = None,
    labels: Optional[jnp.ndarray] = None,
    lora_params: Optional[Dict] = None,
    lora_scale: Optional[float] = None,
) -> jnp.ndarray:
    """Next-token cross entropy with optional MoE load-balancing loss."""
    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    if labels is None:
        labels = input_ids
    labels = jnp.asarray(labels, dtype=jnp.int32)

    out = forward(
        config, params, input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_router_logits=config.output_router_logits,
        lora_params=lora_params,
        lora_scale=lora_scale,
    )
    logits = out["logits"][:, :-1, :]
    target = labels[:, 1:]

    if attention_mask is not None:
        valid = jnp.asarray(attention_mask)[:, 1:].astype(bool)
    else:
        valid = jnp.ones(target.shape, dtype=bool)
    valid = jnp.logical_and(valid, target != IGNORE_INDEX)
    safe_target = jnp.where(valid, target, 0)

    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    token_loss = -jnp.take_along_axis(
        log_probs, safe_target[..., None], axis=-1)[..., 0]
    loss = jnp.sum(jnp.where(valid, token_loss, 0.0)) / jnp.maximum(
        jnp.sum(valid), 1)

    if config.is_moe and config.output_router_logits:
        loss = loss + config.router_aux_loss_coef * out["aux_loss"]
    return loss


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def pad_batch(sequences: Sequence[Sequence[int]], pad_token_id: int = 0,
              max_length: Optional[int] = None
              ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Right-pad a list of token id sequences."""
    sequences = [list(seq) for seq in sequences]
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    if any(len(seq) > max_length for seq in sequences):
        raise ValueError("序列长度超过 max_length")

    input_ids = jnp.full((len(sequences), max_length), pad_token_id, dtype=jnp.int32)
    attention_mask = jnp.zeros((len(sequences), max_length), dtype=jnp.int32)
    for i, seq in enumerate(sequences):
        if not seq:
            continue
        input_ids = input_ids.at[i, :len(seq)].set(jnp.asarray(seq, dtype=jnp.int32))
        attention_mask = attention_mask.at[i, :len(seq)].set(1)
    return input_ids, attention_mask


def _unpack_batch(batch) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    if isinstance(batch, dict):
        ids = batch["input_ids"]
        mask = batch.get("attention_mask")
    else:
        ids = batch[0]
        mask = batch[1] if len(batch) > 1 else None
    ids = jnp.asarray(ids, dtype=jnp.int32)
    mask = None if mask is None else jnp.asarray(mask, dtype=jnp.int32)
    return ids, mask


def make_optimizer(lr: float = 3e-4, weight_decay: float = 0.1,
                   clip_norm: Optional[float] = 1.0) -> optax.GradientTransformation:
    optimizer = optax.adamw(
        learning_rate=lr, b1=0.9, b2=0.95, eps=1e-8,
        weight_decay=weight_decay)
    if clip_norm is not None and clip_norm > 0:
        optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optimizer)
    return optimizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_step(model, batch, opt_state, optimizer: Optional[optax.GradientTransformation] = None,
               lr: float = 3e-4, weight_decay: float = 0.1,
               clip_norm: Optional[float] = 1.0
               ) -> Tuple[float, Dict[str, jnp.ndarray], Any, Dict[str, float]]:
    """One optimizer step; returns ``(loss, grads, opt_state, metrics)``."""
    optimizer = optimizer or make_optimizer(lr, weight_decay, clip_norm)
    input_ids, attention_mask = _unpack_batch(batch)

    if model.lora_params is not None:
        def loss_fn(base, adapters):
            return causal_lm_loss(
                model.config, base, input_ids,
                attention_mask=attention_mask,
                lora_params=adapters,
                lora_scale=model.lora_config.scale)

        loss, grads = jax.value_and_grad(loss_fn, argnums=1)(
            model.params, model.lora_params)
        updates, opt_state = optimizer.update(grads, opt_state, model.lora_params)
        model.lora_params = optax.apply_updates(model.lora_params, updates)
    else:
        def loss_fn(params):
            return causal_lm_loss(model.config, params, input_ids,
                                  attention_mask=attention_mask)

        loss, grads = jax.value_and_grad(loss_fn)(model.params)
        updates, opt_state = optimizer.update(grads, opt_state, model.params)
        model.params = optax.apply_updates(model.params, updates)
    metrics = {
        "loss": float(loss),
        "grad_norm": float(optax.global_norm(grads)),
    }
    return metrics["loss"], grads, opt_state, metrics


def fit(model, batches: Iterable, epochs: int = 1, lr: float = 3e-4,
        weight_decay: float = 0.1, grad_accum: int = 1,
        clip_norm: Optional[float] = 1.0, verbose: bool = False
        ) -> Tuple[Dict[str, jnp.ndarray], List[Dict[str, float]]]:
    """Train ``model.params`` in place and return ``(params, history)``."""
    batches = [_unpack_batch(b) for b in batches]
    if not batches:
        raise ValueError("batches 为空")
    if grad_accum < 1:
        raise ValueError("grad_accum 必须 >= 1")

    optimizer = make_optimizer(lr, weight_decay, clip_norm)
    use_lora = model.lora_params is not None
    params = model.lora_params if use_lora else model.params
    opt_state = optimizer.init(params)
    history: List[Dict[str, float]] = []

    if use_lora:
        def loss_fn(base, adapters, input_ids, attention_mask):
            return causal_lm_loss(
                model.config, base, input_ids,
                attention_mask=attention_mask,
                lora_params=adapters,
                lora_scale=model.lora_config.scale)

        grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=1))

        def compute_grads(adapters, input_ids, attention_mask):
            return grad_fn(model.params, adapters, input_ids, attention_mask)
    else:
        def loss_fn(p, input_ids, attention_mask):
            return causal_lm_loss(model.config, p, input_ids,
                                  attention_mask=attention_mask)

        grad_fn = jax.jit(jax.value_and_grad(loss_fn))

        def compute_grads(p, input_ids, attention_mask):
            return grad_fn(p, input_ids, attention_mask)

    @jax.jit
    def apply_grads(p, state, grads):
        updates, state = optimizer.update(grads, state, p)
        return optax.apply_updates(p, updates), state

    for epoch in range(epochs):
        for start in range(0, len(batches), grad_accum):
            group = batches[start:start + grad_accum]
            total_loss = 0.0
            total_grads = None
            for input_ids, attention_mask in group:
                loss, grads = compute_grads(params, input_ids, attention_mask)
                total_loss += float(loss)
                if total_grads is None:
                    total_grads = grads
                else:
                    total_grads = jax.tree.map(jnp.add, total_grads, grads)
            n = len(group)
            grads = jax.tree.map(lambda g: g / n, total_grads)
            params, opt_state = apply_grads(params, opt_state, grads)
            record = {
                "epoch": epoch,
                "step": len(history),
                "loss": total_loss / n,
                "grad_norm": float(optax.global_norm(grads)),
            }
            history.append(record)
            if verbose:
                print(f"epoch={record['epoch']} step={record['step']} "
                      f"loss={record['loss']:.4f} "
                      f"grad_norm={record['grad_norm']:.4f}")

    return params, history
