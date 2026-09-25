import numpy as np

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# loss function
def cross_entropy(probs, Y):
    eps = 1e-12
    return -np.mean(np.sum(Y * np.log(probs + eps), axis=1))


class NeuralNetwork:
    def __init__(self, layers, lr, batch_size, epochs, rng):
        self.layers = layers
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.rng = rng



    def train(self, X_train, Y_train):

        # How much training data (for batching later)
        n = X_train.shape[0]

        for epoch in range(self.epochs):
            # Permutate the training data 
            perm = self.rng.permutation(n)
            X_train = X_train[perm]
            Y_train = Y_train[perm] 

            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                    
                # Create batches
                end = start + self.batch_size
                X_batch = X_train[start:end]
                Y_batch = Y_train[start:end]

                # Forward pass
                out = X_batch
                for layer in self.layers:
                    out = layer.forward(out)
                probs = softmax(out)

                # Calculate loss
                epoch_loss += cross_entropy(probs, Y_batch)
            
                # Backward pass
                grad = probs - Y_batch
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)

                ##Update
                for layer in self.layers:
                    layer.update(self.lr)
            
            n_batches = (n + self.batch_size - 1) // self.batch_size
            print(f"epoch {epoch}: train loss {epoch_loss / n_batches:.4f}")
                     

    def evaluate(self, X_test, test_labels):

        out = X_test
        for layer in self.layers:
            out = layer.forward(out)
        preds = np.argmax(out, axis=1)
        acc = np.mean(preds == test_labels)
        print(f"Test accuracy {acc:.4f}")

