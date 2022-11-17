# -*- coding: utf-8 -*-
# @Time    : 2022/11/8 15:46
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
from collections import namedtuple
import functools
from functools import partial
import jax.numpy as jnp
from jax._src.util import safe_zip as zip, safe_map as map, unzip2
from jax.tree_util import tree_flatten, tree_unflatten, register_pytree_node

OptimizerState = namedtuple("OptimizerState", ["packed_state", "tree_def", "subtree_defs"])
register_pytree_node(OptimizerState,
                     lambda xs: ((xs.packed_state,), (xs.tree_def, xs.subtree_defs)),
                     lambda data, xs: OptimizerState(xs[0], data[0], data[1]))


class Optimizer:
    def _init(self, params):
        raise NotImplementedError

    def _update(self, step, g, x_batch):
        raise NotImplementedError

    def _get_params(self, params):
        raise NotImplementedError

    def _update_params(self, step, g, x_batch, trainable):
        if trainable: return self._update(step, g, x_batch)
        return x_batch

    def init(self, x0_tree):
        x0_flat, tree = tree_flatten(x0_tree)
        initial_states = [self._init(x0) for x0 in x0_flat]
        states_flat, subtrees = unzip2(map(tree_flatten, initial_states))
        return OptimizerState(states_flat, tree, subtrees)

    def update(self, i, grad_tree, opt_state, trainable):
        states_flat, tree, subtrees = opt_state
        grad_flat, tree2 = tree_flatten(grad_tree)
        if tree2 != tree:
            msg = "优化器更新函数，梯度树和参数树不匹配：参数树 {}  梯度树 {}"
            raise TypeError(msg.format(tree, tree2))
        trainable = trainable if isinstance(trainable, list) else [trainable] * len(grad_flat)
        states = map(tree_unflatten, subtrees, states_flat)
        new_states = map(partial(self._update_params, i), grad_flat, states, trainable)
        new_states_flat, subtrees2 = unzip2(map(tree_flatten, new_states))
        for subtree, subtree2 in zip(subtrees, subtrees2):
            if subtree2 != subtree:
                msg = "优化器更新函数，输出结构和输入结构不匹配: 输入 {}  输出 {}"
                raise TypeError(msg.format(subtree, subtree2))
        return OptimizerState(new_states_flat, tree, subtrees)

    def get_params(self, opt_state):
        states_flat, tree, subtrees = opt_state
        states = map(tree_unflatten, subtrees, states_flat)
        params = map(self._get_params, states)
        return tree_unflatten(tree, params)


class SGD(Optimizer):
    def __init__(self, lr: float):
        self.lr = max(.0, min(lr, 1.0))

    def _init(self, x0):
        return x0

    def _update(self, i, g, x):
        return x - self.lr * g

    def _get_params(self, state):
        return state


class SM3(Optimizer):
    def __init__(self, lr: float, momentum: float = 0.9):
        self.lr = max(.0, min(lr, 1.0))
        self.momentum = momentum

    def splice(self, seq, i, x):
        lst = list(seq)
        lst[i:i + 1] = x
        return lst

    def broadcast_into(self, ndim, x, axis):
        idx = self.splice([None] * ndim, axis, [slice(None)])
        return x[tuple(idx)]

    def _init(self, x0):
        x_shape = x0.shape
        x0 = jnp.atleast_1d(x0)
        vs = [jnp.zeros(sz, dtype=x0.dtype) for sz in x0.shape]
        return x0, jnp.zeros_like(x0), vs, x_shape

    def _update(self, i, g, state):
        x, m, vs, x_shape = state
        vs = [self.broadcast_into(g.ndim, v, i) for i, v in enumerate(vs)]
        accum = functools.reduce(jnp.minimum, vs) + jnp.square(g)
        accum_inv_sqrt = jnp.where(accum > 0, 1. / jnp.sqrt(accum), 0)
        m = (1. - self.momentum) * (g * accum_inv_sqrt) + self.momentum * m
        x = x - self.lr * m
        vs = [accum.max(self.splice(range(x.ndim), j, [])) for j in range(x.ndim)]
        return x, m, vs, x_shape

    def _get_params(self, state):
        x, _, _, x_shape = state
        return x.reshape(x_shape)


class Adagrad(Optimizer):
    def __init__(self, lr: float, momentum: float = 0.9):
        self.lr = max(.0, min(lr, 1.0))
        self.momentum = momentum

    def _init(self, x0):
        g_sq = jnp.zeros_like(x0)
        m = jnp.zeros_like(x0)
        return x0, g_sq, m

    def _update(self, i, g, state):
        x, g_sq, m = state
        g_sq += jnp.square(g)
        g_sq_inv_sqrt = jnp.where(g_sq > 0, 1. / jnp.sqrt(g_sq), 0.0)
        m = (1. - self.momentum) * (g * g_sq_inv_sqrt) + self.momentum * m
        x = x - self.lr * m
        return x, g_sq, m

    def _get_params(self, state):
        x, _, _ = state
        return x


class Adam(Optimizer):
    def __init__(self, lr: float, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = max(.0, min(lr, 1.0))
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def _init(self, x0):
        m0 = jnp.zeros_like(x0)
        v0 = jnp.zeros_like(x0)
        return x0, m0, v0

    def _update(self, i, g, state):
        x, m, v = state
        m = (1 - self.b1) * g + self.b1 * m
        v = (1 - self.b2) * jnp.square(g) + self.b2 * v
        mhat = m / (1 - jnp.asarray(self.b1, m.dtype) ** (i + 1))
        vhat = v / (1 - jnp.asarray(self.b2, m.dtype) ** (i + 1))
        x = x - self.lr * mhat / (jnp.sqrt(vhat) + self.eps)
        return x, m, v

    def _get_params(self, state):
        x, _, _ = state
        return x


class Adamax(Optimizer):
    def __init__(self, lr: float, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = max(.0, min(lr, 1.0))
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def _init(self, x0):
        m0 = jnp.zeros_like(x0)
        u0 = jnp.zeros_like(x0)
        return x0, m0, u0

    def _update(self, i, g, state):
        x, m, u = state
        m = (1 - self.b1) * g + self.b1 * m
        u = jnp.maximum(self.b2 * u, jnp.abs(g))
        x = (x - (self.lr / (1 - jnp.asarray(self.b1, m.dtype) ** (i + 1))) * m
             / (u + self.eps))
        return x, m, u

    def _get_params(self, state):
        x, _, _ = state
        return x


class RMSProp(Optimizer):
    def __init__(self, lr: float, gamma=0.9, eps=1e-8):
        self.lr = max(.0, min(lr, 1.0))
        self.gamma = gamma
        self.eps = eps

    def _init(self, x0):
        avg_sq_grad = jnp.zeros_like(x0)
        return x0, avg_sq_grad

    def _update(self, i, g, state):
        x, avg_sq_grad = state
        avg_sq_grad = avg_sq_grad * self.gamma + jnp.square(g) * (1. - self.gamma)
        x = x - self.lr * g / jnp.sqrt(avg_sq_grad + self.eps)
        return x, avg_sq_grad

    def _get_params(self, state):
        x, _ = state
        return x


class FTRL(Optimizer):
    def __init__(self, lambda_1=.0, lambda_2=.0, beta=1.0, alpha=0.5):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.beta = beta
        self.alpha = alpha

    def _init(self, x0):
        n0 = jnp.zeros_like(x0)
        sigma0 = jnp.zeros_like(x0)
        z0 = jnp.zeros_like(x0)
        return x0, n0, sigma0, z0

    def _update(self, i, g, state):
        x, n, sigma, z = state
        prev_n = n
        n = n + g ** 2
        sigma = (jnp.sqrt(n) - jnp.sqrt(prev_n)) / self.alpha
        z = z + g - sigma * x
        x = jnp.where(jnp.abs(z) >= self.lambda_1,
                      (jnp.sign(z) * self.lambda_1 - z) / ((self.beta + jnp.sqrt(n)) / self.alpha + self.lambda_2),
                      0.0)
        return x, n, sigma, z

    def _get_params(self, state):
        x, _, _, _ = state
        return x


class TG(Optimizer):
    ...


class TGG(Optimizer):
    ...


class FOBOS(Optimizer):
    ...


class RDA(Optimizer):
    ...
