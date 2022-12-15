# -*- coding: utf-8 -*-
# @Time    : 2022/11/26 19:35 
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
import numpy as np
from sunstreaker.initializers import VarianceScaling
from sunstreaker.engine.input_layer import Input
from sunstreaker.engine.functional import Model
from sunstreaker.layers import Conv2D, Embedding, Lambda, AveragePooling2D, UpSampling2D, GroupNormalization, Add, Dense


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
