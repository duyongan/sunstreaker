# -*- coding: utf-8 -*-
# @Time    : 2022/11/8 15:46
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
from sunstreaker.engine.functional import Functional
from sunstreaker.engine.sequential import Sequential
from sunstreaker.engine.training import TrainerSeq


class Model(Sequential):
    def _get_train_cls(self):
        return TrainerSeq

    def _iter_layers(self):
        self._layers = [self]

    def init_in_out(self):
        self._iter_layers()
        self.inputs = self._layers[:1]
        self.outputs = self._layers[-1:]

    def summary(self):
        print("只有一个节点，暂不支持打印子图")

    def plot_model(self, to_file="model.png", dpi=96, rankdir="TB"):
        print("只有一个节点，暂不支持打印子图")


__all__ = [
    "Sequential",
    "Model",
    "Functional",
]
