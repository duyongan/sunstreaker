# -*- coding: utf-8 -*-
# @Time    : 2022/11/8 15:46
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
import jax.numpy as jnp


class Metric:
    def __init__(self, *args, **kwargs):
        ...

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray):
        raise NotImplementedError


class BinaryAccuracy(Metric):
    def __init__(self, acc_thresh, *args, **kwargs):
        super(BinaryAccuracy, self).__init__(*args, **kwargs)
        self.acc_thresh = acc_thresh

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray):
        binary_accuracy(y_true, y_pred, self.acc_thresh)


class Accuracy(Metric):
    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray):
        accuracy(y_true, y_pred)


class CategoricalAccuracy(Metric):
    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray):
        categorical_accuracy(y_true, y_pred)


class SparseCategoricalAccuracy(Metric):
    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray):
        sparse_categorical_accuracy(y_true, y_pred)


class CosineSimilarityAccuracy(Metric):
    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray):
        cosine_similarity_accuracy(y_true, y_pred)


class TopkCategoricalAccuracy(Metric):
    def __init__(self, topk, *args, **kwargs):
        super(TopkCategoricalAccuracy, self).__init__(*args, **kwargs)
        self.topk = topk

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray):
        top_k_categorical_accuracy(y_true, y_pred, self.topk)


class SparseTopkCategoricalAccuracy(Metric):
    def __init__(self, topk, *args, **kwargs):
        super(SparseTopkCategoricalAccuracy, self).__init__(*args, **kwargs)
        self.topk = topk

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray):
        sparse_top_k_categorical_accuracy(y_true, y_pred, self.topk)


def binary_accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray, acc_thresh) -> jnp.ndarray:
    y_pred = jnp.where(y_pred > acc_thresh, 1.0, 0.0)
    return jnp.mean(y_pred == y_true)


def accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(y_true == y_pred)


def categorical_accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    y_pred = jnp.argmax(y_pred, axis=-1)
    y_true = jnp.argmax(y_true, axis=-1)
    return jnp.mean(y_pred == y_true)


def sparse_categorical_accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    y_pred = jnp.argmax(y_pred, axis=-1)
    return jnp.mean(y_pred == y_true)


def cosine_similarity_accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    y_pred = jnp.linalg.norm(y_pred, axis=-1)
    y_true = jnp.linalg.norm(y_true, axis=-1)
    return jnp.sum(y_pred * y_true)


def top_k_categorical_accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray, topk=3) -> jnp.ndarray:
    y_pred = jnp.argmax(y_pred, axis=-1)
    y_true = jnp.argmax(y_true, axis=-1)
    max_k_preds = jnp.argsort(y_pred, axis=-1)[:, -topk:][:, ::-1]
    match_array = jnp.logical_or.reduce(max_k_preds == y_true, axis=-1)
    topk_acc_score = jnp.sum(match_array) / match_array.shape[0]
    return topk_acc_score


def sparse_top_k_categorical_accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray, topk=3) -> jnp.ndarray:
    y_pred = jnp.argmax(y_pred, axis=-1)
    max_k_preds = jnp.argsort(y_pred, axis=-1)[:, -topk:][:, ::-1]
    match_array = jnp.logical_or.reduce(max_k_preds == y_true, axis=-1)
    topk_acc_score = jnp.sum(match_array) / match_array.shape[0]
    return topk_acc_score
