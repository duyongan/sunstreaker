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

    def build(self):
        if self.learn_parameter:
            self.add_weight("gamma", shape=self.input_shape)
            self.add_weight("beta", shape=self.input_shape)
            return self.input_shape
        return self.input_shape

    def call(self, inputs, **kwargs):
        mean = jnp.mean(inputs, axis=self.axis, keepdims=True)
        variance = jnp.var(inputs, axis=self.axis, keepdims=True)
        if not self.learn_parameter:
            outputs = (inputs - mean) / (jnp.sqrt(variance) + self.epsilon)
            return outputs
        gamma, beta = self.get_weight("gamma", "beta")
        if self.trainable:
            mean = self.momentum * mean + (1 - self.momentum) * mean
            variance = self.momentum * variance + (1 - self.momentum) * variance
        outputs = (inputs - mean) / (jnp.sqrt(variance) + self.epsilon)
        if not self.trainable:
            outputs = gamma * outputs + beta
        return outputs


class BatchNormalization(Normalization):
    def __init__(self, **kwargs):
        super().__init__(axis=0, **kwargs)


class LayerNormalization(Normalization):
    def __init__(self, **kwargs):
        super().__init__(axis=tuple(range(len(self.input_shape)))[1:], **kwargs)


class InstanceNormalization(Normalization):
    def __init__(self, **kwargs):
        super().__init__(axis=tuple(range(len(self.input_shape)))[2:], **kwargs)


class GroupNormalization(Normalization):
    def __init__(self, num_groups, **kwargs):
        super().__init__(axis=tuple(range(len(self.input_shape)))[1:], **kwargs)
        self.num_groups = num_groups
        assert self.input_shape[2] % self.num_groups == 0, "通道数必须整除组数"

    def call(self, inputs, **kwargs):
        N, C, H, W = inputs.shape
        inputs = inputs.reshape((N, self.num_groups, C // self.num_groups, H, W))
        outputs = super().call(inputs, **kwargs)
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

    def build(self):
        return self.input_shape

    def call(self, inputs, **kwargs):
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
