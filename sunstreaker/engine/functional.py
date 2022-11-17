# -*- coding: utf-8 -*-
# @Time    : 2022/11/8 17:46 
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
from sunstreaker.engine import training
from typing import List, Union
from ..engine.base_layer import Layer


class Functional(training.ModelBase):
    def __init__(self, inputs: Union[Layer, List[Layer]], outputs: Union[Layer, List[Layer]], **kwargs):
        super().__init__(**kwargs)
        self.inputs = inputs if isinstance(inputs, list) else [inputs]
        self.outputs = outputs if isinstance(outputs, list) else [outputs]
        self._iter_layers()

    def init_in_out(self):
        ...


class Model(Functional):
    ...
