# -*- coding: utf-8 -*-
# @Time    : 2022/11/8 15:46 
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171

from collections import OrderedDict
from jax.random import PRNGKey
import jax.numpy as jnp
from sunstreaker.initializers import GlorotNormal


class Layer:
    count = 0
    seed = 1

    def __init__(self, trainable=True, name=None, **kwargs):
        self._init_set_name(name)
        self.inputs = []
        self.outputs = None
        self.trainable = trainable
        self.params = OrderedDict()
        Layer.count += 1

    def __call__(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        self.inputs = inputs
        self.input_shape = inputs[0].output_shape if isinstance(inputs, list) and len(inputs) == 1 else [_input.output_shape for _input in self.inputs]
        self.output_shape = self.build()
        for k in self.__dict__:
            if isinstance(self.__dict__[k], Layer):
                self.params.update(self.__dict__[k].params)
        return self

    def build(self):
        return self.input_shape

    def call(self, inputs, **kwargs):
        return inputs

    def forward(self, params, inputs, trainable=True, **kwargs):
        inputs = inputs[0] if isinstance(inputs, list) and len(inputs) == 1 else inputs
        self.trainable = trainable
        self.params = params
        self.outputs = self.call(inputs, **kwargs)
        return self.outputs

    def add_weight(self,
                   name,
                   shape=None,
                   dtype=jnp.float32,
                   initializer=GlorotNormal,
                   regularizer=None,
                   constraint=None,
                   **kwargs):
        init = initializer(seed=PRNGKey(Layer.seed))
        Layer.seed += 1
        self.params[self.get_name(name)] = init(shape=shape, dtype=dtype)
        return self.params[self.get_name(name)]

    def _init_set_name(self, name):
        if name is None:
            cls = [cls.__name__.lower() for cls in Layer.__mro__[:-1]]
            self.name = f"{'.'.join(list(reversed(cls)))}_{Layer.count}"
        elif isinstance(name, str):
            self.name = name
        else:
            raise TypeError("参数name不合法")

    def get_name(self, weight_name):
        return f"{self.name}.{weight_name}"

    def get_weight(self, *args):
        if len(args) == 1:
            return self.params[self.get_name(args[0])]
        return (self.params[self.get_name(name)] for name in args)
