import math
import needle as ndl


def rand(*shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False):
    """Generate random numbers uniform between low and high"""
    device = ndl.default_device() if device is None else device
    array = device.rand(*shape, dtype=dtype) * (high - low) + low
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def randn(*shape, mean=0.0, std=1.0, device=None, dtype="float32", requires_grad=False):
    """Generate random normal with specified mean and std deviation"""
    device = ndl.default_device() if device is None else device
    array = device.randn(*shape, dtype=dtype) * std + mean
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    """Generate constant Tensor"""
    device = ndl.default_device() if device is None else device
    array = device.full(shape, c, dtype=dtype)
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def ones(*shape, device=None, dtype="float32", requires_grad=False):
    """Generate all-ones Tensor"""
    return constant(
        *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def zeros(*shape, device=None, dtype="float32", requires_grad=False):
    """Generate all-zeros Tensor"""
    return constant(
        *shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False):
    """Generate binary random Tensor"""
    device = ndl.default_device() if device is None else device
    array = device.rand(*shape) <= p
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def one_hot(n, i, device=None, dtype="float32", requires_grad=False):
    """Generate one-hot encoding Tensor"""
    device = ndl.default_device() if device is None else device
    return ndl.Tensor(
        device.one_hot(n, i.numpy().astype("int32"), dtype=dtype),
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return zeros(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return ones(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def xavier_uniform(fan_in, fan_out, shape=None, gain=1.0, **kwargs):
    shape = shape if shape is not None else (fan_in, fan_out)
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(*shape, low=-a, high=a, **kwargs)


def xavier_normal(fan_in, fan_out, shape=None, gain=1.0, **kwargs):
    shape = shape if shape is not None else (fan_in, fan_out)
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(*shape, std=std, **kwargs)


def kaiming_uniform(fan_in, fan_out=None, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    default_shape = (fan_in,) if fan_out is None else (fan_in, fan_out)
    shape = shape if shape is not None else default_shape
    bound = {"relu": math.sqrt(2)}[nonlinearity] * math.sqrt(3 / fan_in)
    return rand(*shape, low=-bound, high=bound, **kwargs)


def kaiming_normal(fan_in, fan_out=None, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    default_shape = (fan_in,) if fan_out is None else (fan_in, fan_out)
    shape = shape if shape is not None else default_shape
    std = {"relu": math.sqrt(2)}[nonlinearity] / math.sqrt(fan_in)
    return randn(*shape, std=std, **kwargs)
