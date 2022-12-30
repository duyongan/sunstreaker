# -*- coding: utf-8 -*-
# @Time    : 2022/11/28 0:31 
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
import jax.numpy as jnp
from sunstreaker.engine.base_layer import Layer
from jax import lax


class Pool(Layer):
    def __init__(self, kernel_shape=(2, 2), padding_to_same=False, strides=(1, 1), **kwargs):
        super().__init__(**kwargs)
        self.padding = "SAME" if padding_to_same else 'VALID'
        self.strides = strides
        self.kernel_shape = kernel_shape
        self.dividend = jnp.prod(self.kernel_shape)
        self.complete_strides_kernel()

    def complete_strides_kernel(self):
        self.strides = self.strides or (1,) * len(self.kernel_shape)
        self.strides = (1,) + self.strides + (1,)
        self.kernel_shape = (1,) + self.kernel_shape + (1,)

    def pool(self, inputs, init, reduce_fn):
        return lax.reduce_window(inputs, init, reduce_fn, self.kernel_shape, self.strides, self.padding)


class MaxPool2D(Pool):
    def build(self, seed):
        return self.input_shape

    def call(self, inputs, **kwargs):
        return self.pool(inputs, -jnp.inf, lax.max)


class AveragePooling2D(Pool):
    def build(self, seed):
        return self.input_shape

    def call(self, inputs, **kwargs):
        return self.pool(inputs, 0., lax.add) / self.dividend
