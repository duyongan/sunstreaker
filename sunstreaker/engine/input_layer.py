# -*- coding: utf-8 -*-
# @Time    : 2022/11/9 15:56 
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
from .base_layer import Layer


class Input(Layer):
    def __init__(self, input_shape, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape  # tuple 不包括batch size
        self.output_shape = self.input_shape
        self.params = ()
        self.inputs = []
        self.outputs = None
        self.rng = None

    def __call__(self, *args, **kwargs):
        ...

    def call(self, inputs, **kwargs):
        return inputs

    def build(self):
        return self.input_shape
