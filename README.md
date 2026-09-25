# neuralnetwork

A feed-forward neural network for MNIST digit classification, written from scratch
in NumPy — forward pass, backpropagation, mini-batch gradient descent, and training
loop, with no ML framework.

## Architecture

```
input (784) → Linear(784, 512) → ReLU → Linear(512, 10) → softmax
```

- He-initialized linear layers
- ReLU activation
- Softmax output with cross-entropy loss
- Mini-batch SGD with per-epoch shuffling

Reaches ~98.5% test accuracy after 15 epochs.

## Setup

Install dependencies (uses [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
```

Download the MNIST dataset into `data/` (the `data/` directory is gitignored):

```bash
mkdir -p data && cd data
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -LO https://storage.googleapis.com/cvdf-datasets/mnist/$f.gz
done
```

The loaders read the gzipped `.gz` files directly — no manual decompression needed.

## Usage

Train and evaluate:

```bash
uv run python -m neuralnetwork.train
```

This trains on the 60k training images and prints per-epoch loss followed by the
final test accuracy on the 10k held-out test images.

## Project layout

| File | Purpose |
|------|---------|
| `src/neuralnetwork/data.py` | MNIST IDX loading, normalization, one-hot encoding |
| `src/neuralnetwork/layers.py` | `Layer` (linear) and `ReLU` — forward/backward/update |
| `src/neuralnetwork/neuralnetwork.py` | `NeuralNetwork` — softmax, cross-entropy, train/evaluate |
| `src/neuralnetwork/train.py` | Entry point: builds the network and runs training |
