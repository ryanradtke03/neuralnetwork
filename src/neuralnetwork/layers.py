
import numpy as np
rng = np.random.default_rng(42)


class Layer:
    def __init__(self, n_in, n_out, rng):
        self.W = rng.normal(0, np.sqrt(2 / n_in), size=(n_in, n_out))
        self.b = np.zeros((1, n_out))

    # Foward by dot producting in_m with weights then adding bias
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
        

    def backward(self, grad):
        self.dW = self.x.T @ grad
        self.db = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T # dx passing to layer

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask 

    def update(self, lr):
        pass