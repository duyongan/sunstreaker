# -*- coding: utf-8 -*-
# @Time    : 2022/11/8 15:46 
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
from jax import random
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal, normal, ones, zeros


class Layer:
    count = 0

    def __init__(self, trainable=True, name=None, rng=None, **kwargs):
        self._init_set_name(name)
        self.inputs = []
        self.outputs = None
        self.trainable = trainable
        self.params = ()
        self.rng = rng or random.PRNGKey(1)
        Layer.count += 1

    def __call__(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        self.inputs = inputs
        self.input_shape = inputs[0].output_shape if isinstance(inputs, list) and len(inputs) == 1 else [_input.output_shape for _input in self.inputs]
        self.output_shape, self.params = self.build(self.rng)
        return self

    def build(self, rng):
        return self.input_shape, ()

    def call(self, params, inputs, **kwargs):
        return inputs

    def forward(self, params, inputs, trainable=True, **kwargs):
        inputs = inputs[0] if isinstance(inputs, list) and len(inputs) == 1 else inputs
        self.trainable = trainable
        self.outputs = self.call(params, inputs, **kwargs)
        return self.outputs

    def add_weight(self,
                   shape=None,
                   dtype=jnp.float32,
                   rng=None,
                   initializer=glorot_normal,
                   regularizer=None,
                   constraint=None,
                   **kwargs):
        rng = rng or self.rng
        init = initializer(dtype=dtype)
        return init(rng, shape)

    def _init_set_name(self, name):
        if name is None:
            self.name = f"{self.__class__.__name__.lower()}_{Layer.count}"
        elif isinstance(name, str):
            self.name = name
        else:
            raise TypeError("参数name不合法")
