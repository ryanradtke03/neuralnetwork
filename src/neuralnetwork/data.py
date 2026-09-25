import gzip
from pathlib import Path
import numpy as np

# data/ lives at the project root: data.py -> neuralnetwork -> src -> root
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
IMG_MAX_VALUE = 255.0
IMG_SIZE = 784


def one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels]


def load_labels(path):
    """Read an IDX label file (.gz) into a 1-D uint8 array of digits 0-9."""
    with gzip.open(path, "rb") as f:
        header = f.read(8)
        magic = int.from_bytes(header[0:4], "big")
        if magic != 2049:
            raise ValueError(f"{path}: bad label magic {magic}, expected 2049")
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_images(path):
    """Read an IDX image file (.gz) into an (N, 28, 28) uint8 array."""
    with gzip.open(path, "rb") as f:
        header = f.read(16)
        magic = int.from_bytes(header[0:4], "big")
        if magic != 2051:
            raise ValueError(f"{path}: bad image magic {magic}, expected 2051")
        rows = int.from_bytes(header[8:12], "big")
        cols = int.from_bytes(header[12:16], "big")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(-1, rows, cols)


def load_train():
    """Return (images, labels) for the 60k training split."""
    train_images = load_images(DATA_DIR / "train-images-idx3-ubyte.gz")
    train_labels = load_labels(DATA_DIR / "train-labels-idx1-ubyte.gz")
    return train_images, train_labels


def load_test():
    """Return (images, labels) for the 10k test split."""
    test_images = load_images(DATA_DIR / "t10k-images-idx3-ubyte.gz")
    test_labels = load_labels(DATA_DIR / "t10k-labels-idx1-ubyte.gz")
    return test_images, test_labels


def fetch_train():
    train_images, train_labels = load_train()

    # Shape the data 
    imgs = train_images
    imgs = imgs.reshape(-1, IMG_SIZE) # vectorize
    imgs = imgs.astype(np.float32)  # convert to floats for normalization
    imgs = imgs / IMG_MAX_VALUE  # normalize

    X_train = imgs # vectorized images
    Y_train = one_hot(train_labels) # One-hot labels

    return X_train, Y_train, train_images, train_labels
   
def fetch_test():
    test_images, test_labels = load_test()

    # Shape the data
    imgs = test_images
    imgs = imgs.reshape(-1, IMG_SIZE) # vectorize
    imgs = imgs.astype(np.float32)  # convert to floats for normalization
    imgs = imgs / IMG_MAX_VALUE  # normalize

    X_test = imgs # vectorized images
    Y_test = one_hot(test_labels) # One-hot labels

    return X_test, Y_test, test_images, test_labels
