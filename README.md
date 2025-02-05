# Homework 4
Public repository and stub/testing code for Homework 4 of 10-714.

## Dev Environment
- Windows 11
- Miniconda3 (with Python 3.10.6)
- Visual Studio 2022 (with Windows 10 SDK)
- NVCC with CUDA 11.7
- CMake 3.19

## Hints

Part 3: Convolutional neural network
- Convolutional stride should be $1$ in backward process of `ops.Conv`.
- Typo: $6$-th `ConvBN` block in `ResNet9` should be `ConvBN(64, 128, 3, 2)`.
- When copy code from previous homework, pay special attention to each device-relevant function due to ignorance of device option in early works, since implicit data transfer is not allowed.
- `data.Dataloader` should support `device` (and maybe also `dtype`).
- `ndl.cuda()` backend witnesses faster training speed on `CIFAR-10` dataset than `ndl.cpu()`.

Part 4: Recurrent neural network
- In multilayer RNN, hidden state of previous layer `RNNCell[l].h` serves as input to the layer `RNNCell[l+1]`, except for the first layer which consumes word vector from dataset.

Part 5: Long short-term memory network
- Weight (and also bias) of `(i,f,g,o)` is packed into single matrix (4x of size).

Part 6: Penn Treebank dataset
- When clipping raw text to `max_lines` during `Corpus.tokenize()`, it should be **`: max_lines + 1`** in brackets rather than `lines[: max_lines]`. Otherwise, `Corpus` will miss one line and lead to wrong `train_loss` in Part 7.