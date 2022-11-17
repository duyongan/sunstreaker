# -*- coding: utf-8 -*-
# @Time    : 2022/11/8 17:46
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
import dill
import msgpack
import _ctypes
import os
import networkx as nx
from collections import defaultdict
from typing import Callable, Tuple, List, Dict, Optional, Iterable, List, Dict, Any, Sized
from ..engine.base_layer import Layer
from ..optimizers import Optimizer
import pydot_ng as pydot
from prettytable import PrettyTable
from ..engine.input_layer import Input
from sunstreaker.data import convert_to_tensor, pad_sequences, Dataloader
import matplotlib.pyplot as plt
import itertools
from functools import partial
from tqdm import tqdm
import jax.numpy as jnp
from jax import jit, grad, device_put


class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config: Dict[str, Any] = config
        self.mode = "train"
        self.optm = config.get("_optimizer")
        self._layers = config.get("_layers")
        self.outputs = config.get("outputs")
        self.outputs = {out.name: self.outputs for out in self.outputs}
        self._metric_fn = self.config.get("_metrics_fn")
        self._metric_fn = self._metric_fn if isinstance(self._metric_fn, dict) else {k: self._metric_fn for k in self.outputs}
        self._loss_fn = self.config.get("_loss_fn")
        self.forward_pass = self.config.get("forward_fn")
        self.initialize_params = self.config.get("initialize_fn")
        self._loss_fn = self._loss_fn if isinstance(self._loss_fn, dict) else {k: self._loss_fn for k in self.outputs}

    def train(self, data: Dataloader):
        network_params = self.initialize_params(data)
        opt_state = self.optm.init(network_params)
        iter_count = itertools.count()
        progress_bar = tqdm(iterable=range(self.config.get("_epochs")),
                            desc="Training model", leave=True)
        for epoch in progress_bar:
            progress_bar.set_description(desc=f"Epoch {epoch + 1}")
            self.mode = "train"
            for _ in range(data.num_train_batches):
                train_batch = device_put(next(data.train_data))
                opt_state = self.step(next(iter_count), opt_state, train_batch)
                network_params = self.optm.get_params(opt_state)
                self.calculate_metrics(network_params, train_batch)
            network_params = self.optm.get_params(opt_state)
            self.mode = "valid"
            for _ in range(data.num_val_batches):
                valid_batch = device_put(next(data.val_data))
                self.calculate_metrics(network_params, valid_batch)
            self.calculate_epoch_metrics(data)
            progress_bar.set_postfix_str(self.pretty_print_metrics())
            progress_bar.refresh()
        self.config["_trained_params"] = self.optm.get_params(opt_state)
        return self.config

    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch):
        params = self.optm.get_params(opt_state)
        grads = grad(self.compute_loss)(params, batch)
        trainable = [self._layers[i].trainable for i, param in enumerate(params) for _ in param]
        return self.optm.update(i, grads, opt_state, trainable)

    @partial(jit, static_argnums=(0,))
    def compute_loss(self, params, batch):
        inputs, targets = batch
        trainable = True if self.mode == "train" else False
        predictions = self.forward_pass(params, inputs, trainable=trainable)
        loss = 0
        for k in self._loss_fn:
            loss += jit(self._loss_fn[k])(predictions[k], targets[k])
        return loss

    def calculate_metrics(self, params, batch):
        inputs, targets = batch
        trainable = True if self.mode == "train" else False
        predictions = self.forward_pass(params, inputs, trainable=trainable)
        self.config.get("_metrics")["loss"][self.mode].append(self.compute_loss(params, batch))
        for k in self._metric_fn:
            for metric_fn in self._metric_fn[k]:
                self.config.get("_metrics")[metric_fn.__name__][self.mode].append(jit(metric_fn)(predictions[k], targets[k]))

    def calculate_epoch_metrics(self, data: Dataloader):
        self.config.get("_metrics")["loss_per_epoch"]["train"].append(
            jnp.mean(jnp.array(self.config.get("_metrics")["loss"]["train"][-data.num_train_batches:]))
        )
        self.config.get("_metrics")["loss_per_epoch"]["valid"].append(
            jnp.mean(jnp.array(self.config.get("_metrics")["loss"]["valid"][-data.num_val_batches:]))
        )
        for k in self._metric_fn:
            for metric_fn in self._metric_fn[k]:
                self.config.get("_metrics")[metric_fn.__name__ + "_per_epoch"]["train"] \
                    .append(self.config.get("_metrics")[metric_fn.__name__]["train"][-1])
                self.config.get("_metrics")[metric_fn.__name__ + "_per_epoch"]["valid"] \
                    .append(self.config.get("_metrics")[metric_fn.__name__]["valid"][-1])

    def pretty_print_metrics(self) -> str:
        return " :: ".join([f"{metric_type}_{metric_name} : {metric.get(metric_type)[-1]:.3f}"
                            for metric_name, metric in self.config.get("_metrics").items()
                            for metric_type in metric.keys() if "epoch" not in metric_name])


class TrainerSeq(Trainer):
    @partial(jit, static_argnums=(0,))
    def compute_loss(self, params, batch):
        inputs, targets = batch
        trainable = True if self.mode == "train" else False
        predictions = self.forward_pass(params, inputs, trainable=trainable)
        return jit(self.config.get("_loss_fn"))(predictions, targets)

    def calculate_metrics(self, params, batch):
        inputs, targets = batch
        trainable = True if self.mode == "train" else False
        predictions = self.forward_pass(params, inputs, trainable=trainable)
        self.config.get("_metrics")["loss"][self.mode].append(self.compute_loss(params, batch))
        for metric_fn in self.config.get("_metrics_fn"):
            self.config.get("_metrics")[metric_fn.__name__][self.mode].append(jit(metric_fn)(predictions, targets))


class ModelBase(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._layers: List[Layer] = []
        self._epochs: int = 1
        self.input_shape = None
        self._trained_params: List[Optional[Tuple[jnp.ndarray, jnp.ndarray]]] = []
        self._loss_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None
        self._optimizer = None
        self._metrics: Dict[str, Dict[str, List[float]]] = {"loss": defaultdict(list), "loss_per_epoch": defaultdict(list)}
        self._metrics_fn: Optional[List[Callable]] = []
        self._seed: int = 0
        self.graph = nx.DiGraph()
        self.inputs = None
        self.outputs = None
        self.outputs_dict = None
        self.forward_fn = self.forward
        self.initialize_fn = self.initialize_params

    def init_in_out(self):
        self._iter_layers()
        self.outputs_dict = {_layer.name: _layer for _layer in self.outputs}

    def _iter_layers(self):
        def add_edge(ends, G):
            for end in ends:
                if len(end.inputs) <= 0: return
                for layer in end.inputs:
                    G.add_edge(id(end), id(layer))
                add_edge(end.inputs, G)

        add_edge(self.outputs, self.graph)
        id2node = {i + 1: n for i, n in enumerate(self.graph.nodes)}
        node2id = {n: i + 1 for i, n in id2node.items()}
        layers = []
        for start in self.outputs:
            path = list(nx.dfs_tree(self.graph, id(start)))
            layers.append([node2id[n] for n in path])
        graphs = pad_sequences(layers, padding="post")
        graphs = jnp.flipud(graphs.T).tolist()
        rows = []
        nodes = set()
        for graph in graphs:
            row = []
            for node in graph:
                if node != 0:
                    if node not in nodes:
                        row.append(node)
                        nodes.add(node)
            if row: rows.append(row)
        self._layers = [_ctypes.PyObj_FromPtr(id2node[node - 1]) for row in rows for node in row]

    def _init_compile(self, loss):
        self._loss_fn = loss if isinstance(loss, dict) else {out.name: loss for out in self.outputs}

    def compile(self, loss: Callable, optimizer: Optimizer, metrics: List[Callable] = None):
        self.init_in_out()
        self.build()
        self._init_compile(loss)
        self._optimizer = optimizer
        self._metrics_fn = metrics
        for metric_fn in self._metrics_fn:
            self._metrics[metric_fn.__name__] = defaultdict(list)
            self._metrics[metric_fn.__name__ + "_per_epoch"] = defaultdict(list)

    def _get_train_cls(self):
        return Trainer

    def fit(self, data, epochs: int, seed: int = 0):
        assert self._optimizer, "没有传入优化函数"
        assert self._loss_fn, "没有损失函数"
        assert epochs > 0, "epochs要大于0"
        self._epochs = epochs
        self._seed = seed
        self.outputs_dict = {_layer.name: _layer for _layer in self.outputs}
        self.input_shape = {k: data[k].input_shape for k in data} if isinstance(data, Dict) else data.input_shape
        self.__dict__ = self._get_train_cls()(self.__dict__).train(data)

    def build(self, rng=None):
        # input_shape = {k: data[k].input_shape for k in data} if isinstance(data, Dict) else data.input_shape
        trained_params: List[Optional[Tuple[jnp.ndarray, jnp.ndarray]]] = self._trained_params
        if len(trained_params) > 0:
            return [out.output_shape for out in self.outputs], trained_params
        else:
            params = []
            for _layer in self._layers:
                params.append(_layer.params)
            return [out.output_shape for out in self.outputs], params

    def initialize_params(self, data: Dataloader):
        self.output_shape, self.params = self.build()
        return self.params

    def call(self, params, inputs, trainable=True, **kwargs):
        _inputs = inputs
        outputs = {}
        for layer, param in zip(self._layers, params):
            _inputs = [inputs[layer.name]] if isinstance(layer, Input) else [_layer.outputs for _layer in layer.inputs]
            out = layer.forward(param, _inputs, trainable=trainable)
            if layer.name in self.outputs_dict: outputs[layer.name] = out
        return outputs

    def predict(self, inputs: jnp.ndarray):
        assert self._trained_params, "训练后才能预测"
        return self.forward(self._trained_params, inputs, trainable=False)

    def save(self, file_name: str):
        assert self._layers, "Layers是空的"
        assert self._loss_fn, "没有损失函数"
        assert self._metrics_fn, "没有评价函数"
        assert self._optimizer, "没有优化器"
        assert self._trained_params, "模型没有训练"
        self._save_module(file_name, layers=self._layers,
                          loss=self._loss_fn,
                          metrics=self._metrics_fn,
                          optimizer=self._optimizer,
                          params=self._trained_params,
                          input_shape=self.input_shape)

    def load(self, file_name: str):
        deserialized_config = self._load_module(file_name)
        self.input_shape = deserialized_config.get('input_shape')
        self._layers = deserialized_config.get("layers")
        self.compile(loss=deserialized_config.get("loss"),
                     optimizer=deserialized_config.get("optimizer"),
                     metrics=deserialized_config.get("metrics"))
        self._trained_params = convert_to_tensor(deserialized_config.get("params"))
        self.init_in_out()

    def summary(self):
        self.init_in_out()
        table = PrettyTable(['number', 'name', "class", 'input_shape', 'output_shape'], align="l")
        for i, _layer in enumerate(self._layers):
            table.add_row([i, _layer.name, _layer.__class__.__name__, _layer.input_shape, _layer.output_shape])
        print(table)

    def plot_model(self, to_file="model.png", dpi=96, rankdir="TB"):
        """
          rankdir:
          'TB' 竖向排列
          'LR' 横向排列
        """

        def add_edge(dot, src, dst):
            if not dot.get_edge(src, dst):
                dot.add_edge(pydot.Edge(src, dst))

        def path_to_string(path):
            if isinstance(path, os.PathLike):
                return os.fspath(path)
            return path

        self.init_in_out()
        dot = pydot.Dot()
        dot.set("rankdir", rankdir)
        dot.set("concentrate", True)
        dot.set("dpi", dpi)
        dot.set_node_defaults(shape="record")

        for _id in self.graph.nodes:
            _layer = _ctypes.PyObj_FromPtr(_id)
            label = "%s|%s" % (_layer.name, _layer.__class__.__name__)
            label = "{%s}|{input:|output:}|{{%s}|{%s}}" % (
                label,
                _layer.input_shape,
                _layer.output_shape,
            )
            node = pydot.Node(str(_id), label=label)
            dot.add_node(node)
        for _start, _end in self.graph.edges:
            add_edge(dot, _end, _start)

        to_file = path_to_string(to_file)
        _, extension = os.path.splitext(to_file)
        if not extension:
            extension = "png"
        else:
            extension = extension[1:]
        dot.write(to_file, format=extension)

    def _plot(self, train_losses, val_losses, to_file, ylabel):
        epochs = range(1, self._epochs + 1)
        plt.clf()
        plt.plot(epochs, train_losses, color="red", label="Training")
        plt.plot(epochs, val_losses, color="green", label="Validation")
        plt.title(f"{ylabel} Curve")
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.legend()
        # plt.show()
        plt.savefig(to_file)

    def plot_losses(self, to_file="losses.png"):
        train_losses: List[float] = self._metrics.get("loss_per_epoch").get("train")
        val_losses: List[float] = self._metrics.get("loss_per_epoch").get("valid")
        self._plot(train_losses, val_losses, to_file, "Loss")

    def plot_accuracy(self, to_file="accuracy_{}.png"):
        for _metrics in self._metrics.keys():
            if _metrics.endswith("accuracy"):
                train_acc: List[float] = self._metrics.get(f"{_metrics}_per_epoch").get("train")
                val_acc: List[float] = self._metrics.get(f"{_metrics}_per_epoch").get("valid")
                self._plot(train_acc, val_acc, to_file.format(_metrics), "Accuracy")

    @staticmethod
    def _save_module(file_name: str, **config):
        serialized_config: Dict[str, Any] = {}
        for k, v in config.items():
            item_dill: bytes = dill.dumps(v)
            item_msgpack: bytes = msgpack.packb(item_dill, use_bin_type=True)
            serialized_config[k] = item_msgpack

        with open(f"{file_name}.msgpack", "wb") as f:
            serialized_data: bytes = msgpack.packb(serialized_config)
            f.write(serialized_data)

    @staticmethod
    def _load_module(file_name: str) -> Dict[str, Any]:
        with open(f"{file_name}.msgpack", "rb") as f:
            deserialized_data: bytes = f.read()
            deserialized_config: Dict[str, Any] = msgpack.unpackb(deserialized_data)
            for k in list(deserialized_config):
                item_dill: bytes = msgpack.unpackb(deserialized_config.pop(k))
                deserialized_config[k] = dill.loads(item_dill)
        return deserialized_config
