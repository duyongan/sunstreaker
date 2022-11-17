# -*- coding: utf-8 -*-
# @Time    : 2022/11/8 15:46
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
import sys
from jax import lax
import numpy as np
import jax.numpy as jnp
from jax._src import dtypes
from sunstreaker.engine.base_layer import Layer


def deserialize(identifier):
    current_class = getattr(sys.modules[__name__], identifier)
    return current_class


def get(identifier):
    if identifier is None:
        return Linear
    if isinstance(identifier, str):
        identifier = identifier.lower().capitalize()
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError(
            f"Could not interpret activation function identifier: {identifier}"
        )


class Activation(Layer):
    def func(self, x, **kwargs):
        return NotImplementedError

    def build(self, rng):
        return self.input_shape, ()

    def call(self, params, inputs, **kwargs):
        return self.func(inputs)


class Linear(Activation):
    def func(self, x, **kwargs):
        return x


class Softmax(Activation):
    def func(self, x, **kwargs):
        x_max = jnp.max(x, axis=-1, keepdims=True)
        unnormalized = jnp.exp(x - lax.stop_gradient(x_max))
        return unnormalized / jnp.sum(unnormalized, axis=-1, keepdims=True)


class Relu(Activation):
    def func(self, x, **kwargs):
        return jnp.maximum(x, 0)


class Sigmoid(Activation):
    def func(self, x, **kwargs):
        return lax.logistic(x)


class Elu(Activation):
    def func(self, x, **kwargs):
        alpha = 1.0
        safe_x = jnp.where(x > 0, 0., x)
        return jnp.where(x > 0, x, alpha * jnp.expm1(safe_x))


class LeakyRelu(Activation):
    def func(self, x, **kwargs):
        negative_slope = 1e-2
        return jnp.where(x >= 0, x, negative_slope * x)


class Leakyrelu(LeakyRelu):
    ...


class Gelu(Activation):
    def func(self, x, approximate=True, **kwargs):
        x = x.astype(dtypes.to_inexact_dtype(x.dtype))

        if approximate:
            sqrt_2_over_pi = np.sqrt(2 / np.pi).astype(x.dtype)
            cdf = 0.5 * (1.0 + jnp.tanh(sqrt_2_over_pi * (x + 0.044715 * (x ** 3))))
            return x * cdf
        else:
            sqrt_2 = np.sqrt(2).astype(x.dtype)
            return jnp.array(x * (lax.erf(x / sqrt_2) + 1) / 2, dtype=x.dtype)
