# -*- coding: utf-8 -*-
# @Time    : 2022/11/26 19:35 
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
import numpy as np
import jax.numpy as jnp
from sunstreaker.engine.input_layer import Input
from sunstreaker.engine.functional import Model
from sunstreaker.layers import Conv2D, Embedding, Lambda, AveragePooling2D, UpSampling2D, GroupNormalization, Add, Dense


def _compute_fans(shape, data_format='channels_last'):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) in {3, 4, 5}:
        if data_format == 'channels_first':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif data_format == 'channels_last':
            receptive_field_size = np.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise ValueError('Invalid data_format: ' + data_format)
    else:
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out


class Initializer(object):
    """Initializer base class: all initializers inherit from this class.
    """

    def __call__(self, shape, dtype=None):
        raise NotImplementedError

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        if 'dtype' in config:
            config.pop('dtype')
        return cls(**config)


class VarianceScaling(Initializer):
    def __init__(self, scale=1.0,
                 mode='fan_in',
                 distribution='normal',
                 seed=None):
        if scale <= 0.:
            raise ValueError('`scale` must be a positive float. Got:', scale)
        mode = mode.lower()
        if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
            raise ValueError('Invalid `mode` argument: '
                             'expected on of {"fan_in", "fan_out", "fan_avg"} '
                             'but got', mode)
        distribution = distribution.lower()
        if distribution not in {'normal', 'uniform'}:
            raise ValueError('Invalid `distribution` argument: '
                             'expected one of {"normal", "uniform"} '
                             'but got', distribution)
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    def __call__(self, shape, dtype=None):
        fan_in, fan_out = _compute_fans(shape)
        scale = self.scale
        if self.mode == 'fan_in':
            scale /= max(1., fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1., fan_out)
        else:
            scale /= max(1., float(fan_in + fan_out) / 2)
        if self.distribution == 'normal':
            # 0.879... = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = np.sqrt(scale) / .87962566103423978
            x = K.truncated_normal(shape, 0., stddev,
                                   dtype=dtype, seed=self.seed)
        else:
            limit = np.sqrt(3. * scale)
            x = K.random_uniform(shape, -limit, limit,
                                 dtype=dtype, seed=self.seed)
        if self.seed is not None:
            self.seed += 1
        return x

    def get_config(self):
        return {
            'scale': self.scale,
            'mode': self.mode,
            'distribution': self.distribution,
            'seed': self.seed
        }


def dense(x, out_dim, activation=None, init_scale=1):
    """Dense包装
    """
    init_scale = max(init_scale, 1e-10)
    initializer = VarianceScaling(init_scale, 'fan_avg', 'uniform')
    return Dense(
        out_dim,
        activation=activation,
        use_bias=False,
        kernel_initializer=initializer
    )(x)


def residual_block(x, ch, t):
    """残差block
    """
    in_dim = K.int_shape(x)[-1]
    out_dim = ch * embedding_size
    if in_dim == out_dim:
        xi = x
    else:
        xi = dense(x, out_dim)
    x = Add()([x, dense(t, K.int_shape(x)[-1])])
    x = conv2d(x, out_dim, 'swish', 1 / num_layers ** 0.5)
    x = conv2d(x, out_dim, 'swish', 1 / num_layers ** 0.5)
    x = Add()([x, xi])
    x = GroupNormalization()(x)
    return x


def conv2d(x, out_dim, activation=None, init_scale=1):
    """Conv2D包装
    """
    init_scale = max(init_scale, 1e-10)
    initializer = VarianceScaling(init_scale, 'fan_avg', 'uniform')
    return Conv2D(
        out_dim, (3, 3),
        padding='same',
        activation=activation,
        use_bias=False,
        kernel_initializer=initializer
    )(x)


# 基本配置
imgs = list_pictures('celeba-64/', 'jpg')
# imgs += list_pictures('celeba-64/', 'jpg')
np.random.shuffle(imgs)
img_size = 128  # 如果只想快速实验，可以改为64
batch_size = 64  # 如果显存不够，可以降低为32、16，但不建议低于16
embedding_size = 128
channels = [1, 1, 2, 2, 4, 4]
num_layers = len(channels) * 2 + 1
blocks = 2  # 如果显存不够，可以降低为1
min_pixel = 4  # 不建议降低，显存足够可以增加到8

# 超参数选择
T = 1000
alpha = np.sqrt(1 - 0.02 * np.arange(1, T + 1) / T)
beta = np.sqrt(1 - alpha ** 2)
bar_alpha = np.cumprod(alpha)
bar_beta = np.sqrt(1 - bar_alpha ** 2)
sigma = beta.copy()

x_in = x = Input(shape=(img_size, img_size, 3))
t_in = Input(shape=(1,))
t = Embedding(input_dim=T, output_dim=embedding_size)(t_in)
t = Lambda(lambda t: t[:, None])(t)

x = conv2d(x, embedding_size)
inputs, skip_pooling = [x], 0

for i, ch in enumerate(channels):
    for j in range(blocks):
        x = residual_block(x, ch, t)
        inputs.append(x)
    if min(K.int_shape(x)[1:3]) > min_pixel:
        x = AveragePooling2D((2, 2))(x)
        inputs.append(x)
    else:
        skip_pooling += 1

x = residual_block(x, ch, t)
inputs.pop()

for i, ch in enumerate(channels[::-1]):
    if i >= skip_pooling:
        x = UpSampling2D((2, 2))(x)
        x = Add()([x, inputs.pop()])
    for j in range(blocks):
        xi = inputs.pop()
        x = residual_block(x, K.int_shape(xi)[-1] // embedding_size, t)
        x = Add()([x, xi])

x = GroupNormalization()(x)
x = conv2d(x, 3)

model = Model(inputs=[x_in, t_in], outputs=x)
model.summary()
