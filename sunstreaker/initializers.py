# -*- coding: utf-8 -*-
# @Time    : 2022/11/29 14:14 
# @Author  : duyongan
# @Email   : 13261051171@163.com
# @phone   : 13261051171
import sys
from jax import random
import jax.numpy as jnp


def deserialize(identifier):
    current_class = getattr(sys.modules[__name__], identifier)
    return current_class


def get(identifier):
    if isinstance(identifier, str):
        identifier = identifier.lower().capitalize()
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError(f"没有这种初始化方式: {identifier}")


class Initializer:
    def __call__(self, shape, dtype=jnp.float32):
        raise NotImplementedError


class Zeros(Initializer):
    def __call__(self, shape, dtype=jnp.float32):
        return jnp.zeros(shape, dtype)


class Ones(Initializer):
    def __call__(self, shape, dtype=jnp.float32):
        return jnp.ones(shape, dtype)


class Constant(Initializer):
    def __init__(self, value=0):
        self.value = value

    def __call__(self, shape, dtype=jnp.float32):
        return jnp.ones(shape, dtype) * self.value


class RandomNormal(Initializer):
    def __init__(self, seed, mean=0., stddev=0.05):
        self.mu = mean
        self.sigma = stddev
        self.seed = seed

    def __call__(self, shape, dtype=jnp.float32):
        return self.mu + self.sigma * random.normal(key=self.seed, shape=shape, dtype=dtype)


class RandomUniform(Initializer):
    def __init__(self, seed, minval=-0.05, maxval=0.05):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape, dtype=jnp.float32):
        return random.uniform(key=self.seed, shape=shape, dtype=dtype, minval=self.minval, maxval=self.maxval)


class TruncatedNormal(Initializer):
    def __init__(self, seed, mean=0., stddev=0.05, lower=-2, upper=2):
        self.mu = mean
        self.sigma = stddev
        self.lower = lower + self.mu
        self.upper = upper + self.mu
        self.seed = seed

    def __call__(self, shape, dtype=jnp.float32):
        # 0.879... = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        sigma = self.sigma / jnp.array(.87962566103423978, dtype)
        return self.mu + sigma * random.truncated_normal(key=self.seed, lower=self.lower, upper=self.upper, shape=shape, dtype=dtype)


class Identity(Initializer):
    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, shape, dtype=jnp.float32):
        return self.gain * jnp.eye(*shape)


class Orthogonal(Initializer):
    def __init__(self, seed, gain=1.0):
        self.seed = seed
        self.gain = gain

    def __call__(self, shape, dtype=jnp.float32):
        num_rows, num_cols = jnp.prod(shape[:-1]), shape[-1]
        flat_shape = (max(num_cols, num_rows), min(num_cols, num_rows))
        A = random.normal(self.seed, flat_shape, dtype)
        Q, R = jnp.linalg.qr(A)
        d = jnp.sign(jnp.diag(R))
        Q *= d  # norm的正负概率是均等的，可以使得Q uniform
        if num_rows < num_cols: Q = Q.T
        return self.gain * jnp.reshape(Q, shape)


class DeltaOrthogonal(Orthogonal):
    def __call__(self, shape, dtype=jnp.float32):
        w = super.__call__(shape, dtype)
        "待实现"
        ...


class VarianceScaling(Initializer):
    def __init__(self, seed, scale=1.0, mode='fan_in', distribution='normal'):
        assert scale <= 0, "scale不能小于0"
        assert mode in {'fan_in', 'fan_out', 'fan_avg'}, "没有这种模式"
        assert distribution in {'normal', 'uniform', 'truncated_normal'}, "没有这种分布"
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    @staticmethod
    def _compute_fans(shape):
        assert len(shape) > 1, f"shape：{shape}错误"
        match len(shape):
            case 1:
                fan_in = fan_out = shape[0]
            case 2:
                fan_in, fan_out = shape[0], shape[1]
            case _:
                receptive_field_size = int(jnp.prod(shape[:-2]))
                fan_in = shape[-2] * receptive_field_size
                fan_out = shape[-1] * receptive_field_size
        return fan_in, fan_out

    def __call__(self, shape, dtype=jnp.float32):
        fan_in, fan_out = self._compute_fans(shape)

        match self.mode:
            case "fan_in":
                denominator = max(1.0, fan_in)
            case "fan_out":
                denominator = max(1.0, fan_out)
            case "fan_avg":
                denominator = max(1.0, (fan_in + fan_out) / 2.0)
            case _:
                raise ValueError("没有这种模式")

        variance = jnp.array(self.scale / denominator, dtype=dtype)

        match self.distribution:
            case "truncated_normal":
                stddev = jnp.sqrt(variance) / jnp.array(.87962566103423978, dtype)
                return random.truncated_normal(self.seed, -2, 2, shape, dtype) * stddev
            case "normal":
                return random.normal(self.seed, shape, dtype) * jnp.sqrt(variance)
            case "uniform":
                return random.uniform(self.seed, shape, dtype, -1) * jnp.sqrt(3 * variance)
            case _:
                raise ValueError("没有这种分布")


class LecunUniform(VarianceScaling):
    def __init__(self, seed=None):
        super().__init__(scale=1.0, mode="fan_in", distribution="uniform", seed=seed)


class LecunNormal(VarianceScaling):
    def __init__(self, seed=None):
        super().__init__(scale=1.0, mode="fan_in", distribution="truncated_normal", seed=seed)


class HeNormal(VarianceScaling):
    def __init__(self, seed=None):
        super().__init__(scale=2.0, mode="fan_in", distribution="truncated_normal", seed=seed)


class HeUniform(VarianceScaling):
    def __init__(self, seed=None):
        super().__init__(scale=2.0, mode="fan_in", distribution="uniform", seed=seed)


class GlorotNormal(VarianceScaling):
    def __init__(self, seed=None):
        super().__init__(scale=1.0, mode="fan_avg", distribution="truncated_normal", seed=seed)


class GlorotUniform(VarianceScaling):
    def __init__(self, seed=None):
        super().__init__(scale=1.0, mode="fan_avg", distribution="uniform", seed=seed)


XavierUniform = GlorotUniform
XavierNormal = GlorotNormal
KaimingUniform = HeUniform
KaimingNormal = HeNormal
