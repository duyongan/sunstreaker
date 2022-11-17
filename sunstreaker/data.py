# -*- coding: utf-8 -*-
# @Time    : 2022/11/8 15:46
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
from jax import device_put

import numpy as np
from typing import Iterator, Tuple, Union, Dict


class Dataloader:
    def __init__(self,
                 train_data: Union[Iterator[Tuple[np.ndarray, np.ndarray]], Iterator[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]],
                 val_data: Union[Iterator[Tuple[np.ndarray, np.ndarray]], Iterator[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]],
                 input_shape: Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]],
                 batch_size: int,
                 num_train_batches: int,
                 num_val_batches: int
                 ):
        self.train_data = train_data
        self.val_data = val_data
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_train_batches = num_train_batches
        self.num_val_batches = num_val_batches


def convert_to_tensor(data):
    flat_data, data_tree_struct = tree_flatten(data)
    for i, item in enumerate(flat_data):
        flat_data[i] = device_put(item)
    return tree_unflatten(data_tree_struct, flat_data)


def pad_sequences(
        sequences,
        maxlen=None,
        dtype="int32",
        padding="pre",
        truncating="pre",
        value=0.0,
):
    if not hasattr(sequences, "__len__"):
        raise ValueError("sequences必须有__len__属性")
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True
    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = jnp.asarray(x, dtype=jnp.int64).shape[1:]
                flag = False
        except TypeError as e:
            raise ValueError(
                "sequences list包含的元素必须有__len__属性 "
                f"找到不可iterable的元素: {str(x)}"
            ) from e

    if maxlen is None:
        maxlen = jnp.max(jnp.array(lengths))

    x = jnp.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue
        if truncating == "pre":
            trunc = s[-maxlen:]
        elif truncating == "post":
            trunc = s[:maxlen]
        else:
            raise ValueError(f'截断类型【{truncating}】不存在')

        trunc = jnp.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                f"Shape of sample {trunc.shape[1:]} of sequence at "
                f"position {idx} is different from expected shape "
                f"{sample_shape}"
            )

        if padding == "post":
            x = x.at[idx, : len(trunc)].set(trunc)
        elif padding == "pre":
            x = x.at[idx, -len(trunc):].set(trunc)
        else:
            raise ValueError(f'Padding类型【{padding}】不存在')
    return x
