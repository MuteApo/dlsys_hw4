"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import math


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features), device=device, dtype=dtype
        )
        self.bias = Parameter(
            init.kaiming_uniform(out_features, 1).transpose(), device=device, dtype=dtype
        )

    def forward(self, X: Tensor) -> Tensor:
        out = X @ self.weight
        if self.has_bias:
            out += self.bias.broadcast_to(out.shape)
        return out


class Flatten(Module):
    def forward(self, X):
        return ops.flatten(X)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return (1 + ops.exp(-x)) ** (-1)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.modules:
            x = layer(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        lse = ops.logsumexp(logits, 1)
        y_one_hot = init.one_hot(logits.shape[-1], y, device=y.device, dtype=y.dtype)
        zy = (logits * y_one_hot).sum(1)
        return (lse - zy).sum() / logits.shape[0]


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(dim), device=device, dtype=dtype)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        ex, dx = self.norm_train(x) if self.training else self.norm_test(x)
        w = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        b = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        return w * (x - ex) / (dx + self.eps) ** 0.5 + b

    def norm_train(self, x: Tensor) -> Tensor:
        ex = x.sum(0) / x.shape[0]
        self.running_mean = (
            (1 - self.momentum) * self.running_mean + self.momentum * ex
        ).detach()
        ex = ex.reshape((1, self.dim)).broadcast_to(x.shape)
        dx = ((x - ex) ** 2).sum(0) / x.shape[0]
        self.running_var = (
            (1 - self.momentum) * self.running_var + self.momentum * dx
        ).detach()
        dx = dx.reshape((1, self.dim)).broadcast_to(x.shape)
        return ex, dx

    def norm_test(self, x: Tensor) -> Tensor:
        ex = self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)
        dx = self.running_var.reshape((1, self.dim)).broadcast_to(x.shape)
        return ex, dx


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2, 3)).transpose((1, 2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.has_bias = bias
        self.weight = Parameter(
            init.kaiming_uniform(
                in_channels * kernel_size**2,
                shape=(kernel_size, kernel_size, in_channels, out_channels),
            ),
            device=device,
            dtype=dtype,
        )
        bound = 1 / math.sqrt(in_channels * kernel_size**2)
        self.bias = Parameter(
            init.rand(out_channels, low=-bound, high=bound), device=device, dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        out = ops.conv(
            x.transpose((1, 2)).transpose((2, 3)),
            self.weight,
            self.stride,
            self.kernel_size // 2,
        )
        if self.has_bias:
            out += self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(
                (*out.shape[:-1], self.out_channels)
            )
        return out.transpose((2, 3)).transpose((1, 2))


class RNNCell(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.has_bias = bias
        self.activation = {"tanh": Tanh(), "relu": ReLU()}[nonlinearity]
        bound = 1 / math.sqrt(hidden_size)
        self.W_ih = Parameter(
            init.rand(input_size, hidden_size, low=-bound, high=bound),
            device=device,
            dtype=dtype,
        )
        self.W_hh = Parameter(
            init.rand(hidden_size, hidden_size, low=-bound, high=bound),
            device=device,
            dtype=dtype,
        )
        self.bias_ih = Parameter(
            init.rand(hidden_size, low=-bound, high=bound), device=device, dtype=dtype
        )
        self.bias_hh = Parameter(
            init.rand(hidden_size, low=-bound, high=bound), device=device, dtype=dtype
        )

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        if h is None:
            h = init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype)
        out = X @ self.W_ih + h @ self.W_hh
        if self.has_bias:
            out += self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(out.shape)
            out += self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(out.shape)
        return self.activation(out)


class RNN(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cells = [
            RNNCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                bias,
                nonlinearity,
                device=device,
                dtype=dtype,
            )
            for i in range(num_layers)
        ]

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        X_split = ops.split(X, 0)
        if h0 is None:
            h0 = init.zeros(
                self.num_layers,
                X.shape[1],
                self.hidden_size,
                device=X.device,
                dtype=X.dtype,
            )
        h_split = list(ops.split(h0, 0))
        out = []
        for i in range(X.shape[0]):
            for j in range(self.num_layers):
                h_split[j] = self.rnn_cells[j](
                    X_split[i] if j == 0 else h_split[j - 1], h_split[j]
                )
            out += [h_split[-1]]
        return ops.stack(out, 0), ops.stack(h_split, 0)


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.has_bias = bias
        bound = 1 / math.sqrt(hidden_size)
        self.W_ih = Parameter(
            init.rand(input_size, 4 * hidden_size, low=-bound, high=bound),
            device=device,
            dtype=dtype,
        )
        self.W_hh = Parameter(
            init.rand(hidden_size, 4 * hidden_size, low=-bound, high=bound),
            device=device,
            dtype=dtype,
        )
        self.bias_ih = Parameter(
            init.rand(4 * hidden_size, low=-bound, high=bound), device=device, dtype=dtype
        )
        self.bias_hh = Parameter(
            init.rand(4 * hidden_size, low=-bound, high=bound), device=device, dtype=dtype
        )
        self.activation = [Sigmoid(), Sigmoid(), Tanh(), Sigmoid()]

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        if h is None:
            h = (
                init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype),
            ) * 2
        h0, c0 = h
        out = X @ self.W_ih + h0 @ self.W_hh
        if self.has_bias:
            out += self.bias_ih.reshape((1, 4 * self.hidden_size)).broadcast_to(out.shape)
            out += self.bias_hh.reshape((1, 4 * self.hidden_size)).broadcast_to(out.shape)
        out = ops.split(ops.reshape(out, (X.shape[0], 4, self.hidden_size)), 1)
        i, f, g, o = [x(out[i]) for i, x in enumerate(self.activation)]
        c_next = f * c0 + i * g
        h_next = o * ops.tanh(c_next)
        return h_next, c_next


class LSTM(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = [
            LSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                bias,
                device=device,
                dtype=dtype,
            )
            for i in range(num_layers)
        ]

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        X_split = ops.split(X, 0)
        if h is None:
            h = (
                init.zeros(
                    self.num_layers,
                    X.shape[1],
                    self.hidden_size,
                    device=X.device,
                    dtype=X.dtype,
                ),
            ) * 2
        h_split, c_split = [list(ops.split(_, 0)) for _ in h]
        out = []
        for i in range(X.shape[0]):
            for j in range(self.num_layers):
                h_split[j], c_split[j] = self.lstm_cells[j](
                    X_split[i] if j == 0 else h_split[j - 1], (h_split[j], c_split[j])
                )
            out += [h_split[-1]]
        return ops.stack(out, 0), (ops.stack(h_split, 0), ops.stack(c_split, 0))


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            init.randn(num_embeddings, embedding_dim), device=device, dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        seq_len, batch_size = x.shape
        x_one_hot = init.one_hot(
            self.num_embeddings,
            x.reshape((seq_len * batch_size,)),
            device=x.device,
            dtype=x.dtype,
        )
        return (x_one_hot @ self.weight).reshape(
            (seq_len, batch_size, self.embedding_dim)
        )
