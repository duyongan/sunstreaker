# -*- coding: utf-8 -*-
# @Time    : 2022/11/8 15:46
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
import functools
import operator as op
import jax.numpy as jnp
from jax import random
from sunstreaker.engine.base_layer import Layer
from sunstreaker import activations
from jax.nn import sigmoid
from jax.nn.initializers import glorot_normal, normal
from functools import partial
from jax import lax


class Dense(Layer):
    def __init__(self, units, activation=None, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.use_bias = use_bias
        self.activation = activations.get(activation)()
        self.units = int(units) if not isinstance(units, int) else units

    def build(self, rng):
        k1, k2 = random.split(rng)
        output_shape = self.input_shape[:-1] + (self.units,)
        self.kernel = self.add_weight((self.input_shape[-1], self.units), initializer=glorot_normal, rng=k1)
        if self.use_bias:
            self.bias = self.add_weight((self.units,), initializer=normal, rng=k2)
            return output_shape, (self.kernel, self.bias)
        else:
            return output_shape, (self.kernel,)

    def call(self, params, inputs, **kwargs):
        if self.use_bias:
            self.kernel, self.bias = params
            outputs = jnp.dot(inputs, self.kernel) + self.bias
        else:
            self.kernel, = params
            outputs = jnp.dot(inputs, self.kernel)
        outputs = self.activation.forward(params=None, inputs=outputs)
        return outputs


class Flatten(Layer):
    def build(self, rng):
        output_shape = functools.reduce(op.mul, self.input_shape, 1),
        return output_shape, ()

    def call(self, params, inputs, **kwargs):
        outputs = jnp.reshape(inputs, (inputs.shape[0], -1))
        return outputs


class Dropout(Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def build(self, rng):
        return self.input_shape, ()

    def call(self, params, inputs, **kwargs):
        rng = kwargs.get('rng', None)
        if rng is None:
            msg = "缺少随机参数rng"
            raise ValueError(msg)
        if self.trainable:
            keep = random.bernoulli(rng, self.rate, inputs.shape)
            return jnp.where(keep, inputs / self.rate, 0)
        else:
            return inputs


class Conv2D(Layer):
    def __init__(self, kernel_shape=(2, 2), padding="SAME", strides=(1, 1), **kwargs):
        super().__init__(**kwargs)
        # 'VALID'
        self.padding = padding
        self.strides = strides
        self.kernel_shape = kernel_shape

    def build(self, rng):
        output_shape = self.input_shape
        self.W = self.add_weight((self.input_shape[-1], self.kernel_shape), initializer=glorot_normal, rng=rng)
        return output_shape, (self.W,)

    def call(self, params, inputs, **kwargs):
        self.W, = params
        out = lax.conv(jnp.transpose(inputs, [0, 3, 1, 2]),
                       jnp.transpose(self.W, [3, 2, 0, 1]),
                       self.strides,
                       self.padding)
        return out


class Pool(Layer):
    def __init__(self, kernel_shape=(2, 2), padding="SAME", strides=(1, 1), **kwargs):
        super().__init__(**kwargs)
        # 'VALID'
        self.padding = padding
        self.strides = strides
        self.kernel_shape = kernel_shape

    def pool(self, inputs, init, reduce_fn):
        strides = self.strides or (1,) * len(self.kernel_shape)
        strides = (1,) + strides + (1,)
        dims = (1,) + self.kernel_shape + (1,)

        is_single_input = False
        if inputs.ndim == len(dims) - 1:
            inputs = inputs[None]
            is_single_input = True

        y = lax.reduce_window(inputs, init, reduce_fn, dims, strides, self.padding)
        if is_single_input:
            y = jnp.squeeze(y, axis=0)
        return y


class MaxPool2D(Pool):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def build(self, rng):
        return self.input_shape, ()

    def call(self, params, inputs, **kwargs):
        y = self.pool(inputs, -jnp.inf, lax.max)
        return y


class AvgPool2D(Pool):
    def __init__(self, count_include_pad=True, **kwargs):
        super().__init__(**kwargs)
        self.count_include_pad = count_include_pad

    def build(self, rng):
        return self.input_shape, ()

    def call(self, params, inputs, **kwargs):
        y = self.pool(inputs, 0., lax.add)
        if self.count_include_pad:
            y = y / jnp.prod(self.kernel_shape)
        else:
            div_shape = inputs.shape[:-1] + (1,)
            if len(div_shape) - 2 == len(self.kernel_shape):
                div_shape = (1,) + div_shape[1:]
            y = y / self.pool(jnp.ones(div_shape), 0., lax.add)
        return y


class GRU(Layer):
    def __init__(self, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim

    def build(self, rng):
        W_init, b_init = glorot_normal(), normal()
        hidden = b_init(rng, (self.input_shape[0], self.out_dim))

        k1, k2, k3 = random.split(rng, num=3)
        update_W, update_U, update_b = (
            W_init(k1, (self.input_shape[2], self.out_dim)),
            W_init(k2, (self.out_dim, self.out_dim)),
            b_init(k3, (self.out_dim,)),)

        k1, k2, k3 = random.split(rng, num=3)
        reset_W, reset_U, reset_b = (
            W_init(k1, (self.input_shape[2], self.out_dim)),
            W_init(k2, (self.out_dim, self.out_dim)),
            b_init(k3, (self.out_dim,)),)

        k1, k2, k3 = random.split(rng, num=3)
        out_W, out_U, out_b = (
            W_init(k1, (self.input_shape[2], self.out_dim)),
            W_init(k2, (self.out_dim, self.out_dim)),
            b_init(k3, (self.out_dim,)),)

        output_shape = (self.input_shape[0], self.input_shape[1], self.out_dim)
        return (output_shape,
                (hidden,
                 (update_W, update_U, update_b),
                 (reset_W, reset_U, reset_b),
                 (out_W, out_U, out_b),),)

    @staticmethod
    def cell(params, hidden, inp):
        _, (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (out_W, out_U, out_b) = params
        update_gate = sigmoid(jnp.dot(inp, update_W) + jnp.dot(hidden, update_U) + update_b)
        reset_gate = sigmoid(jnp.dot(inp, reset_W) + jnp.dot(hidden, reset_U) + reset_b)
        output_gate = jnp.tanh(jnp.dot(inp, out_W) + jnp.dot(jnp.multiply(reset_gate, hidden), out_U) + out_b)
        output = jnp.multiply(update_gate, hidden) + jnp.multiply(1 - update_gate, output_gate)
        hidden = output
        return hidden, hidden

    def call(self, params, inputs, **kwargs):
        h = params[0]
        inputs = jnp.moveaxis(inputs, 1, 0)
        f = partial(self.cell, params)
        _, h_new = lax.scan(f, h, inputs)
        return h_new
