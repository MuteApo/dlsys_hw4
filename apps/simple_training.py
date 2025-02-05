import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
from models import *
import time

device = ndl.cpu()

### CIFAR-10 training ###


def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    model.train()
    if opt is None:
        model.eval()

    total_acc = total_loss = 0
    for _, (X, y) in enumerate(dataloader):
        pred = model(X)
        total_acc += np.sum(pred.numpy().argmax(1) == y.numpy())

        loss = loss_fn(pred, y)
        total_loss += loss.numpy().squeeze() * y.shape[0]

        if opt is not None:
            loss.backward()
            opt.step()
            opt.reset_grad()

    avg_acc = total_acc / len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_acc, avg_loss


def train_cifar10(
    model,
    dataloader,
    n_epochs=1,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    loss_fn=nn.SoftmaxLoss,
):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)

    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for i in range(n_epochs):
        start = time.perf_counter()
        avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn(), opt)
        end = time.perf_counter()
        print(
            f"train_epoch_{i} in {end - start:.3f}s: train_acc={avg_acc:.6f} train_loss={avg_loss:.6f}"
        )

    return avg_acc, avg_loss


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    start = time.perf_counter()
    avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn())
    end = time.perf_counter()
    print(
        f"eval_epoch in {end - start:.3f}s: eval_acc={avg_acc:.6f} eval_loss={avg_loss:.6f}"
    )

    return avg_acc, avg_loss


### PTB training ###
def epoch_general_ptb(
    data,
    model,
    seq_len=40,
    loss_fn=nn.SoftmaxLoss(),
    opt=None,
    clip=None,
    device=None,
    dtype="float32",
):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    model.train()
    if opt is None:
        model.eval()

    h = None
    n_sample = total_acc = total_loss = 0
    for i in range(0, data.shape[0] - 1, seq_len):
        X, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
        n_sample += y.shape[0]
        pred, h_next = model(X, h)
        h = [_.detach() for _ in h_next] if isinstance(h_next, tuple) else h_next.detach()
        total_acc += np.sum(pred.numpy().argmax(1) == y.numpy())

        loss = loss_fn(pred, y)
        total_loss += loss.numpy().squeeze() * y.shape[0]

        if opt is not None:
            loss.backward()
            if clip is not None:
                opt.clip_grad_norm(clip)
            opt.step()
            opt.reset_grad()

    avg_acc = total_acc / n_sample
    avg_loss = total_loss / n_sample
    return avg_acc, avg_loss


def train_ptb(
    model,
    data,
    seq_len=40,
    n_epochs=1,
    optimizer=ndl.optim.SGD,
    lr=4.0,
    weight_decay=0.0,
    loss_fn=nn.SoftmaxLoss,
    clip=None,
    device=None,
    dtype="float32",
):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)

    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for i in range(n_epochs):
        start = time.perf_counter()
        avg_acc, avg_loss = epoch_general_ptb(
            data, model, seq_len, loss_fn(), opt, clip, device=device, dtype=dtype
        )
        end = time.perf_counter()
        print(
            f"train_epoch_{i} in {end - start:.3f}s: train_acc={avg_acc:.6f} train_loss={avg_loss:.6f}"
        )

    return avg_acc, avg_loss


def evaluate_ptb(
    model, data, seq_len=40, loss_fn=nn.SoftmaxLoss, device=None, dtype="float32"
):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    start = time.perf_counter()
    avg_acc, avg_loss = epoch_general_ptb(
        data, model, seq_len, loss_fn(), device=device, dtype=dtype
    )
    end = time.perf_counter()
    print(
        f"eval_epoch in {end - start:.3f}s: eval_acc={avg_acc:.6f} eval_loss={avg_loss:.6f}"
    )

    return avg_acc, avg_loss


if __name__ == "__main__":
    ### For testing purposes
    device = ndl.cpu()
    # dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    # dataloader = ndl.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True
    #         )
    #
    # model = ResNet9(device=device, dtype="float32")
    # train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
    #      lr=0.001, weight_decay=0.001)

    corpus = ndl.data.Corpus("./data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = ndl.data.batchify(
        corpus.train, batch_size, device=device, dtype="float32"
    )
    model = LanguageModel(
        1, len(corpus.dictionary), hidden_size, num_layers=2, device=device
    )
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
