# -*- coding: utf-8 -*-
# @Time    : 2022/11/8 15:46
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
import jax
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


class Embedding(Layer):
    def __init__(self, vocabulary_size, dimension, **kwargs):
        super().__init__(**kwargs)
        self.vocabulary_size = vocabulary_size
        self.dimension = dimension

    def build(self, rng):
        self.embeddings = self.add_weight((self.vocabulary_size, self.dimension), initializer=normal)
        return self.input_shape[:-1] + (self.dimension,), self.embeddings

    def call(self, params, inputs, **kwargs):
        self.embeddings, = params
        return self.embeddings[inputs]


class Lambda(Layer):
    def __init__(self, function, **kwargs):
        super().__init__(**kwargs)
        self.function = function

    def build(self, rng):
        ...

    def call(self, params, inputs, **kwargs):
        return self.function(inputs)


class Add(Layer):
    def __init__(self, function, **kwargs):
        super().__init__(**kwargs)
        self.function = function

    def build(self, rng):
        ...

    def call(self, params, inputs, **kwargs):
        return jnp.sum(inputs, axis=-1)


class Concatenate(Layer):
    def __init__(self, function, **kwargs):
        super().__init__(**kwargs)
        self.function = function

    def build(self, rng):
        ...

    def call(self, params, inputs, **kwargs):
        return jnp.concatenate(inputs, axis=-1)


class Dot(Layer):
    def __init__(self, normalize=False, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize

    def build(self, rng):
        return self.input_shape[0][:-1] + self.input_shape[0][-1:], ()

    def call(self, params, inputs, **kwargs):
        assert len(inputs) == 2, "Dot输入只能是两个"
        x1 = inputs[0]
        x2 = inputs[1]
        y = jnp.dot(x1, x2)
        return y


class Multiply(Layer):
    def build(self, rng):
        return self.input_shape, ()

    def call(self, params, inputs, **kwargs):
        assert len(inputs) == 2, "Multiply输入只能是两个"
        x1 = inputs[0]
        x2 = inputs[1]
        y = x1 * x2
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


class Conv2D(Layer):
    def __init__(self, kernel_shape=(2, 2), padding_to_same=False, strides=(1, 1), activation=None, use_bias=True, initializer=normal, **kwargs):
        super().__init__(**kwargs)
        self.padding = "SAME" if padding_to_same else 'VALID'
        self.strides = strides
        self.use_bias = use_bias
        self.activation = activations.get(activation)()
        self.kernel_shape = kernel_shape
        self.initializer = initializer

    def build(self, rng):
        k1, k2 = random.split(rng)
        output_shape = self.input_shape
        self.kernel = self.add_weight((self.input_shape[-1], self.kernel_shape), initializer=glorot_normal, rng=k1)
        if self.use_bias:
            self.bias = self.add_weight((self.kernel,), initializer=self.initializer, rng=k2)
            return output_shape, (self.kernel, self.bias)
        else:
            return output_shape, (self.kernel,)

    def call(self, params, inputs, **kwargs):
        if self.use_bias:
            self.kernel, self.bias = params
        else:
            self.kernel, = params
        outputs = lax.conv(jnp.transpose(inputs, [0, 3, 1, 2]),
                           jnp.transpose(self.kernel, [3, 2, 0, 1]),
                           self.strides,
                           self.padding)
        if self.use_bias:
            outputs = outputs + self.bias
        return outputs


class UpSampling2D(Layer):
    def __init__(self, interpolation='bilinear', times=2, **kwargs):
        super().__init__(**kwargs)
        self.interpolation = interpolation
        self.times = times

    def build(self, rng):
        return self.input_shape, ()

    def call(self, params, inputs, **kwargs):
        B, H, W, C = inputs.shape
        return jax.image.resize(inputs, shape=(B, H * self.times, W * self.times, C), method=self.interpolation)
