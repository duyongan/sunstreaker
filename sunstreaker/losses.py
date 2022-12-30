# -*- coding: utf-8 -*-
# @Time    : 2022/11/8 15:46
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
import jax.numpy as jnp


class Loss:
    def __init__(self, *args, **kwargs):
        ...

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray):
        raise NotImplementedError


class BinaryCrossentropy(Loss):
    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray):
        return binary_crossentropy(y_true, y_pred)


class CategoricalCrossentropy(Loss):
    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray):
        return categorical_crossentropy(y_true, y_pred)


class MeanSquaredError(Loss):
    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray):
        return mean_squared_error(y_true, y_pred)


class MeanAbsoluteError(Loss):
    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray):
        return mean_absolute_error(y_true, y_pred)


class MeanSquaredLogarithmicError(Loss):
    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray):
        return mean_squared_logarithmic_error(y_true, y_pred)


class Hinge(Loss):
    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray):
        return hinge(y_true, y_pred)


class KlDivergence(Loss):
    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray):
        return kl_divergence(y_true, y_pred)


class Huber(Loss):
    def __init__(self, delta=1.0, *args, **kwargs):
        super(Huber, self).__init__(*args, **kwargs)
        self.delta = delta

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray):
        return huber(y_true, y_pred, self.delta)


class L2Error(Loss):
    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray):
        return l2_error(y_true, y_pred)


def huber(y_true: jnp.ndarray, y_pred: jnp.ndarray, delta=1.0) -> jnp.ndarray:
    x = y_true - y_pred
    loss = jnp.where(x > delta, delta * jnp.abs(x) - 0.5 * jnp.square(delta), 0.5 * jnp.square(x))
    return -jnp.mean(loss, axis=jnp.arange(len(y_pred.shape)))


def binary_crossentropy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return -jnp.mean((y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred)), axis=jnp.arange(len(y_pred.shape)))


def categorical_crossentropy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return -jnp.mean(jnp.sum(y_pred * y_true, axis=jnp.arange(len(y_pred.shape))))


def mean_squared_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return -jnp.mean(jnp.square(y_pred - y_true), axis=jnp.arange(len(y_pred.shape)))


def mean_absolute_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return -jnp.mean(jnp.absolute(y_pred - y_true), axis=jnp.arange(len(y_pred.shape)))


def mean_squared_logarithmic_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return -jnp.sum(jnp.square(jnp.log(y_pred + 1) - jnp.log(y_true + 1)), axis=jnp.arange(len(y_pred.shape)))


def hinge(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return -jnp.mean(jnp.max(jnp.log(1 - y_pred * y_true), 0), axis=jnp.arange(len(y_pred.shape)))


def kl_divergence(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return -jnp.sum(y_true * jnp.log(y_true / y_pred), axis=jnp.arange(len(y_pred.shape)))


def l2_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return -jnp.sum(jnp.square(y_pred - y_true), axis=jnp.arange(len(y_pred.shape)))
