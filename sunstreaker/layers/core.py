# -*- coding: utf-8 -*-
# @Time    : 2022/11/8 15:46
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
import jax
from typing import Optional
import functools
import operator as op
import jax.numpy as jnp
from jax import random
from sunstreaker.engine.base_layer import Layer
from sunstreaker import activations, initializers
from jax.nn import sigmoid
from sunstreaker.initializers import GlorotUniform, Zeros
from jax import lax


class Dense(Layer):
    def __init__(self, units, activation=None, use_bias=True, kernel_initializer=GlorotUniform, bias_initializer=Zeros, **kwargs):
        super().__init__(**kwargs)
        self.use_bias = use_bias
        self.activation = activations.get(activation)()
        self.units = int(units) if not isinstance(units, int) else units
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self):
        output_shape = self.input_shape[:-1] + (self.units,)
        self.add_weight("kernel", (self.input_shape[-1], self.units), initializer=self.kernel_initializer)
        if self.use_bias:
            self.add_weight("bias", (self.units,), initializer=self.bias_initializer)
        return output_shape

    def call(self, inputs, **kwargs):
        kernel = self.get_weight("kernel")
        if self.use_bias:
            bias = self.get_weight("bias")
            outputs = jnp.dot(inputs, kernel) + bias
        else:
            outputs = jnp.dot(inputs, kernel)
        outputs = self.activation.forward(params=None, inputs=outputs)
        return outputs


class Flatten(Layer):
    def build(self):
        output_shape = functools.reduce(op.mul, self.input_shape, 1),
        return output_shape

    def call(self, inputs, **kwargs):
        outputs = jnp.reshape(inputs, (inputs.shape[0], -1))
        return outputs


class Dropout(Layer):
    def __init__(self, rate, deterministic=False, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.deterministic: Optional[bool] = deterministic

    def build(self):
        return self.input_shape

    def call(self, inputs, **kwargs):
        if self.rate == 0.:
            return inputs
        if self.rate == 1.0:
            return jnp.zeros_like(inputs)
        if self.deterministic:
            return inputs
        seed = kwargs.get('seed', None)
        if seed is None:
            msg = "缺少随机参数rng"
            raise ValueError(msg)
        if self.trainable:
            keep_prob = 1. - self.rate
            keep = random.bernoulli(seed, keep_prob, inputs.shape)
            return jnp.where(keep, inputs / keep_prob, 0)
        else:
            return inputs


class Embedding(Layer):
    def __init__(self,
                 vocabulary_size,
                 dimension,
                 initializer='zeros',
                 **kwargs):
        super().__init__(**kwargs)
        self.vocabulary_size = vocabulary_size
        self.dimension = dimension
        self.initializer = initializers.get(initializer) if isinstance(initializer, str) else initializer

    def build(self):
        self.add_weight("embedding", (self.vocabulary_size, self.dimension), initializer=self.initializer)
        return self.input_shape[:-1] + (self.dimension,)

    def call(self, inputs, **kwargs):
        embeddings = self.get_weight("embedding")
        position_ids = jnp.array(inputs, dtype=jnp.int32)
        return embeddings[position_ids]


class Lambda(Layer):
    def __init__(self, function, **kwargs):
        super().__init__(**kwargs)
        self.function = function

    def call(self, inputs, **kwargs):
        return self.function(inputs)


class Add(Layer):
    def call(self, inputs, **kwargs):
        return jnp.sum(inputs, axis=-1)


class Concatenate(Layer):
    def call(self, inputs, **kwargs):
        return jnp.concatenate(inputs, axis=-1)


class Dot(Layer):
    def __init__(self, normalize=False, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize

    def build(self):
        return self.input_shape[0][:-1] + self.input_shape[0][-1:]

    def call(self, inputs, **kwargs):
        assert len(inputs) == 2, "Dot输入只能是两个"
        x1 = inputs[0]
        x2 = inputs[1]
        y = jnp.dot(x1, x2)
        return y


class Multiply(Layer):
    def build(self):
        return self.input_shape

    def call(self, inputs, **kwargs):
        assert len(inputs) == 2, "Multiply输入只能是两个"
        x1 = inputs[0]
        x2 = inputs[1]
        y = x1 * x2
        return y


class GRU(Layer):
    def __init__(self, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim

    def build(self):
        w_init = functools.partial(self.add_weight, shape=(self.input_shape[2], self.out_dim), initializer=GlorotUniform)
        u_init = functools.partial(self.add_weight, shape=(self.out_dim, self.out_dim), initializer=GlorotUniform)
        b_init = functools.partial(self.add_weight, shape=(self.out_dim,), initializer=Zeros)
        self.add_weight("hidden", (self.input_shape[0], self.out_dim), initializer=GlorotUniform)
        w_init("update_W"), u_init("update_U"), b_init("update_b")
        w_init("reset_W"), u_init("reset_U"), b_init("reset_b")
        w_init("out_W"), u_init("out_U"), b_init("out_b")
        output_shape = (self.input_shape[0], self.input_shape[1], self.out_dim)
        return output_shape

    def cell(self, hidden, inp):
        update_W, update_U, update_b, reset_W, reset_U, reset_b, out_W, out_U, out_b = self.get_weight("update_W", "update_U", "update_U", "reset_W", "reset_U", "reset_b", "out_W", "out_U", "out_b")
        update_gate = sigmoid(jnp.dot(inp, update_W) + jnp.dot(hidden, update_U) + update_b)
        reset_gate = sigmoid(jnp.dot(inp, reset_W) + jnp.dot(hidden, reset_U) + reset_b)
        output_gate = jnp.tanh(jnp.dot(inp, out_W) + jnp.dot(jnp.multiply(reset_gate, hidden), out_U) + out_b)
        output = jnp.multiply(update_gate, hidden) + jnp.multiply(1 - update_gate, output_gate)
        hidden = output
        return hidden, hidden

    def call(self, inputs, **kwargs):
        h = self.params[0]
        inputs = jnp.moveaxis(inputs, 1, 0)
        f = functools.partial(self.cell, self.params)
        _, h_new = lax.scan(f, h, inputs)
        return h_new


class Conv2D(Layer):
    def __init__(self, kernel_shape=(2, 2), padding_to_same=False, strides=(1, 1), activation=None, use_bias=True, initializer=GlorotUniform, **kwargs):
        super().__init__(**kwargs)
        self.padding = "SAME" if padding_to_same else 'VALID'
        self.strides = strides
        self.use_bias = use_bias
        self.activation = activations.get(activation)()
        self.kernel_shape = kernel_shape
        self.initializer = initializer

    def build(self):
        output_shape = self.input_shape
        self.add_weight("kernel", (self.input_shape[-1], self.kernel_shape), initializer=GlorotUniform())
        if self.use_bias:
            self.add_weight("bias", (self.kernel_shape,), initializer=self.initializer)
        return output_shape

    def call(self, inputs, **kwargs):
        kernel = self.get_weight("kernel")
        outputs = lax.conv(jnp.transpose(inputs, [0, 3, 1, 2]),
                           jnp.transpose(kernel, [3, 2, 0, 1]),
                           self.strides,
                           self.padding)
        if self.use_bias:
            outputs += self.get_weight("bias")
        return outputs


class UpSampling2D(Layer):
    def __init__(self, interpolation='bilinear', times=2, **kwargs):
        super().__init__(**kwargs)
        self.interpolation = interpolation
        self.times = times

    def build(self):
        return self.input_shape

    def call(self, inputs, **kwargs):
        B, H, W, C = inputs.shape
        return jax.image.resize(inputs, shape=(B, H * self.times, W * self.times, C), method=self.interpolation)
