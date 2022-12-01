"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
# from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * self.scalar


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a**self.scalar

    def gradient(self, out_grad, node):
        return out_grad * self.scalar * node.inputs[0] ** (self.scalar - 1)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / rhs ** 2


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        axes = self.axes if self.axes is not None else (a.ndim - 2, a.ndim - 1)
        order = tuple(sum(axes) - i if i in axes else i for i in range(a.ndim))
        return a.permute(order)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.reshape(self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.broadcast_to(self.shape).compact()

    def gradient(self, out_grad, node):
        in_shape = node.inputs[0].shape
        axes = tuple(i for i in range(len(in_shape)) if in_shape[i] != self.shape[i])
        return reshape(summation(out_grad, axes), in_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = (axes,) if isinstance(axes, int) else axes
        if self.axes is not None:
            self.axes = tuple(sorted(self.axes, reverse=True))

    def compute(self, a):
        if self.axes is None:
            out = a.sum()
        else:
            out = a
            for _, x in enumerate(self.axes):
                out = out.sum((x,))
        return out

    def gradient(self, out_grad, node):
        in_shape = node.inputs[0].shape
        axes = self.axes if self.axes is not None else range(len(in_shape))
        out_shape = [1 if i in axes else in_shape[i] for i in range(len(in_shape))]
        return broadcast_to(reshape(out_grad, tuple(out_shape)), in_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        grad_1 = out_grad @ rhs.transpose()
        grad_2 = lhs.transpose() @ out_grad
        batch_1 = len(grad_1.shape) - len(lhs.shape)
        batch_2 = len(grad_2.shape) - len(rhs.shape)
        if batch_1 > 0:
            grad_1 = grad_1.sum(tuple(range(batch_1)))
        if batch_2 > 0:
            grad_2 = grad_2.sum(tuple(range(batch_2)))
        return grad_1, grad_2


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        input = node.inputs[0]
        return out_grad * Tensor(
            input.realize_cached_data() > 0, device=input.device, dtype=input.dtype
        )


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = (axes,) if isinstance(axes, int) else axes

    def compute(self, Z):
        max_z = Z.max(self.axes, keepdims=True)
        exp_z = array_api.exp(Z - max_z.broadcast_to(Z.shape))
        sum_z = exp_z.sum(self.axes)
        log_z = array_api.log(sum_z)
        return log_z + max_z.reshape(log_z.shape)

    def gradient(self, out_grad, node):
        Z = node.inputs[0].realize_cached_data()
        max_z = Z.max(self.axes, keepdims=True)
        exp_z = array_api.exp(Z - max_z.broadcast_to(Z.shape))
        sum_z = exp_z.sum(self.axes)
        axes = self.axes if self.axes is not None else range(len(Z.shape))
        shape = [1 if i in axes else Z.shape[i] for i in range(len(Z.shape))]
        return reshape(out_grad / sum_z, shape).broadcast_to(Z.shape) * exp_z


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        return out_grad * (1 - tanh(node.inputs[0]) ** 2)


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        stacked = array_api.empty((len(args), args[0].size), dtype=args[0].dtype, device=args[0].device)
        for i, t in enumerate(args):
            stacked[i, :] = t.reshape((1, t.size))
        stacked = stacked.reshape((len(args), *args[0].shape))
        order = list(range(1, args[0].ndim + 1))
        order.insert(self.axis, 0)
        return stacked.permute(order)

    def gradient(self, out_grad, node):
        return split(out_grad, self.axis)


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        order = list(range(A.ndim))
        order.remove(self.axis)
        order.insert(0, self.axis)
        A_p = A.permute(order)
        count = A.shape[self.axis]
        A_r = A_p.reshape((count, A.size // count))
        return [A_r[i, :].reshape(A_p.shape[1:]) for i in range(count)]

    def gradient(self, out_grad, node):
        return stack(out_grad, self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return a.flip(self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes)


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        scaler = tuple(self.dilation + 1 if i in self.axes else 1 for i in range(a.ndim))
        shape = tuple(x * scaler[i] for i, x in enumerate(a.shape))
        pos = tuple(slice(0, shape[i], scaler[i]) for i in range(a.ndim))
        out = array_api.full(shape, 0, dtype=a.dtype, device=a.device)
        out[pos] = a
        return out

    def gradient(self, out_grad, node):
        return undilate(out_grad, self.axes, self.dilation)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        scaler = tuple(self.dilation + 1 if i in self.axes else 1 for i in range(a.ndim))
        pos = tuple(slice(0, a.shape[i], scaler[i]) for i in range(a.ndim))
        return a[pos].compact()

    def gradient(self, out_grad, node):
        return dilate(out_grad, self.axes, self.dilation)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        A_p = A.pad(((0, 0), (self.padding,) * 2, (self.padding,) * 2, (0, 0)))
        N, H, W, Ci = A_p.shape
        Ns, Hs, Ws, Cs = A_p.strides
        K, _, _, Co = B.shape
        Ho, Wo = (H - K) // self.stride + 1, (W - K) // self.stride + 1
        A_2d = A_p.as_strided(
            (N, Ho, Wo, K, K, Ci), (Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)
        ).reshape((N * Ho * Wo, K * K * Ci))
        B_2d = B.reshape((K * K * Ci, Co))
        return (A_2d @ B_2d).reshape((N, Ho, Wo, Co))

    def gradient(self, out_grad, node):
        x, w = node.inputs
        w = flip(w, (0, 1)).transpose((2, 3))  # (K, K, Co, Ci)
        g = dilate(out_grad, (1, 2), self.stride - 1)  # (N, H-K+1, W-K+1, Co)
        grad_1 = conv(g, w, 1, w.shape[0] - self.padding - 1)
        x = transpose(x, (0, 3))  # (Ci, H, W, N)
        g = g.transpose((0, 1)).transpose((1, 2))  # (H-K+1, W-K+1, N, Co)
        grad_2 = conv(x, g, 1, self.padding).transpose((0, 1)).transpose((1, 2))
        return grad_1, grad_2


def conv(a, b, stride=1, padding=0):
    return Conv(stride, padding)(a, b)


class Flatten(TensorOp):
    def compute(self, a):
        return a.reshape((a.shape[0], a.size // a.shape[0]))

    def gradient(self, out_grad, node):
        input = node.inputs[0]
        return reshape(out_grad, input.shape)


def flatten(a):
    return Flatten()(a)
