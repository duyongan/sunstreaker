# -*- coding: utf-8 -*-
# @Time    : 2022/11/8 15:46
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
from typing import List
from ..engine.base_layer import Layer
from sunstreaker.engine import training


class Sequential(training.ModelBase):
    def __init__(self, layers=None, **kwargs):
        super().__init__(inputs=None, outputs=None)
        if layers is None:
            layers = []
        self._layers: List[Layer] = layers

    def __add__(self, other):
        assert type(other) == Sequential, "两个Sequential才能相加"
        assert len(other._layers) > 0, "Layers是空的"
        layers = self._layers + other._layers
        return Sequential(layers=layers)

    def add(self, other):
        if isinstance(other, Sequential) and len(other._layers) > 0:
            self._layers += other._layers
        elif isinstance(other, list) and len(other) > 0:
            self._layers += other
        else:
            return None

    def _get_train_cls(self):
        return training.TrainerSeq

    def init_in_out(self):
        assert len(self._layers) > 0, "Layers是空的"
        self.inputs = self._layers[:1]
        self.outputs = self._layers[-1:]
        self._iter_layers()
        layers = [_layer(self._layers[i - 1]) for i, _layer in enumerate(self._layers) if i > 0]
        layers.insert(0, self._layers[0])
        self._layers = layers

    def _iter_layers(self):
        for i in range(len(self._layers) - 1):
            self.graph.add_edge(id(self._layers[i + 1]), id(self._layers[i]))

    def _init_compile(self, loss):
        self._loss_fn = loss

    def call(self, params, inputs, trainable=True, **kwargs):
        for layer, param in zip(self._layers, params):
            inputs = layer.forward(param, inputs, trainable=trainable)
        return inputs


class Model(Sequential):
    ...
