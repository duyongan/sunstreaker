# -*- coding: utf-8 -*-
# @Time    : 2022/12/16 15:32 
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
import jax.numpy as jnp
from sunstreaker import initializers
from sunstreaker.layers import Dense, Embedding
from sunstreaker.activations import Softmax
from sunstreaker.engine.base_layer import Layer


def sequence_masking(x, mask, value=-jnp.inf, axis=-1):
    if mask is None:
        return x

    x_dtype = jnp.dtype(x)
    if x_dtype == 'bool':
        x = x.astype(jnp.float32)
    if axis is None:
        axis = 1
    elif axis < 0:
        axis = jnp.ndim(x) + axis  # 2
    assert axis > 0, 'axis越界了'
    x.at[1 - mask].set(value)
    if x_dtype == 'bool':
        x = jnp.array(x, bool)
    return x


class MultiHeadAttention(Layer):
    def __init__(self,
                 heads,
                 head_size,
                 out_dim=None,
                 key_size=None,
                 use_bias=True,
                 normalization='softmax',
                 attention_scale=True,
                 attention_dropout=None,
                 return_attention_scores=False,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = out_dim or heads * head_size
        self.key_size = key_size or head_size
        self.use_bias = use_bias
        self.normalization = normalization
        self.attention_scale = attention_scale
        self.attention_dropout = attention_dropout
        self.return_attention_scores = return_attention_scores
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self):
        self.q_dense = Dense(units=self.key_size * self.heads,
                             use_bias=self.use_bias,
                             kernel_initializer=self.kernel_initializer
                             )
        self.k_dense = Dense(units=self.key_size * self.heads,
                             use_bias=self.use_bias,
                             kernel_initializer=self.kernel_initializer
                             )
        self.v_dense = Dense(units=self.key_size * self.heads,
                             use_bias=self.use_bias,
                             kernel_initializer=self.kernel_initializer
                             )
        self.o_dense = Dense(units=self.key_size * self.heads,
                             use_bias=self.use_bias,
                             kernel_initializer=self.kernel_initializer
                             )
        self.params.update(self.q_dense.params)
        self.params.update(self.k_dense.params)
        self.params.update(self.v_dense.params)
        self.params.update(self.o_dense.params)
        return self.input_shape[0][:2] + (self.out_dim,)

    def call(self, inputs, mask=None, **kwargs):
        q, k, v = inputs[:3]
        q_mask, v_mask = None, None
        if mask is not None:
            q_mask, v_mask = mask[0], mask[2]
        qw = self.q_dense.forward(self.params, q)
        kw = self.k_dense.forward(self.params, k)
        vw = self.v_dense.forward(self.params, v)
        qw = jnp.reshape(qw, (self.heads, self.key_size), -1)
        kw = jnp.reshape(kw, (self.heads, self.key_size), -1)
        vw = jnp.reshape(vw, (self.heads, self.key_size), -1)
        a = jnp.einsum('bjhd,bkhd->bhjk', qw, kw)
        a = sequence_masking(a, v_mask)
        A = Softmax().func(a)
        o = jnp.einsum('bhjk,bkhd->bjhd', A, vw)
        o = o.reshape(o.shape[:-2] + (jnp.prod(o.shape[-2:])))
        o = self.o_dense.forward(self.params, o)
        return o
