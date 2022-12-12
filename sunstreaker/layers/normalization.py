# -*- coding: utf-8 -*-
# @Time    : 2022/11/27 17:06 
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
from jax import lax
import jax.numpy as jnp
from sunstreaker.engine.base_layer import Layer


class Normalization(Layer):
    def __init__(self, axis, momentum=0.99, epsilon=1e-3, learn_parameter=False, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.learn_parameter = learn_parameter

    def build(self, rng):
        if self.learn_parameter:
            self.gamma = self.add_weight(shape=self.input_shape)
            self.beta = self.add_weight(shape=self.input_shape)
            return self.input_shape, (self.gamma, self.beta)
        return self.input_shape, ()

    def call(self, params, inputs, **kwargs):
        mean = jnp.mean(inputs, axis=self.axis, keepdims=True)
        variance = jnp.var(inputs, axis=self.axis, keepdims=True)
        if self.trainable and self.learn_parameter:
            self.gamma, self.beta = params
            mean = self.momentum * mean + (1 - self.momentum) * mean
            variance = self.momentum * variance + (1 - self.momentum) * variance
        outputs = (inputs - mean) / (jnp.sqrt(variance) + self.epsilon)
        if not self.trainable and self.learn_parameter:
            outputs = self.gamma * outputs + self.beta
        return outputs


class BatchNormalization(Normalization):
    def __init__(self, **kwargs):
        super().__init__(axis=0, **kwargs)


class LayerNormalization(Normalization):
    def __init__(self, **kwargs):
        super().__init__(axis=(1, 2, 3), **kwargs)


class InstanceNormalization(Normalization):
    def __init__(self, **kwargs):
        super().__init__(axis=(2, 3), **kwargs)


class GroupNormalization(Normalization):
    def __init__(self, num_groups, **kwargs):
        super().__init__(axis=(2, 3, 4), **kwargs)
        self.num_groups = num_groups
        assert self.input_shape[2] % self.num_groups == 0, "通道数必须整除组数"

    def call(self, params, inputs, **kwargs):
        N, C, H, W = inputs.shape
        inputs = inputs.reshape((N, self.num_groups, C // self.num_groups, H, W))
        outputs = super().call(params, inputs, **kwargs)
        return outputs.reshape((N, C, H, W))


class LocalResponseNormalization(Layer):
    """AlexNet
    https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    """

    def __init__(self, group_size=5, k=2, alpha=1e-4, beta=0.75, **kwargs):
        super().__init__(**kwargs)
        self.n = group_size
        self.k = k
        self.alpha = alpha
        self.beta = beta

    def build(self, rng):
        return self.input_shape, ()

    def call(self, params, inputs, **kwargs):
        y = lax.reduce_window(inputs, 0., lax.add, (1, 1, 1, self.n), (1, 1, 1, 1), "VALID") / self.n
        prefix = jnp.tile(y[:1], (self.n // 2,) + (1,) * len(self.input_shape))
        y = jnp.concatenate([prefix, y])
        if y.shape[-1] < inputs.shape[-1]:
            suffix = jnp.flip(y[:, :, :, -self.n], axis=-1)
            d = jnp.arange(self.n)[None, None, None, :] + 1
            suffix = jnp.flip(jnp.cumsum(suffix, axis=-1) / d, axis=-1)
            y = jnp.concatenate([y, suffix])[:, :, :, :inputs.shape[-1]]
        y = inputs / jnp.power(self.k + self.alpha * jnp.square(y), self.beta)
        return y
