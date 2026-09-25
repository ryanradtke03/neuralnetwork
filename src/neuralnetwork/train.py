import numpy as np
from neuralnetwork.layers import Layer, ReLU
from neuralnetwork.data import fetch_train, fetch_test
from neuralnetwork.neuralnetwork import NeuralNetwork


def main():
    # Consts
    input_layer_size = 784
    layer_1_size = 512
    output_layer_size = 10
    rng = np.random.default_rng(42) # randomize this later

    ## Architecture
    layers = [
      Layer(input_layer_size, layer_1_size, rng),
      ReLU(),
      Layer(layer_1_size, output_layer_size, rng),
    ]
    
    # Hyperparameters
    lr = 0.01
    batch_size = 32
    epochs = 15

    # Data
    X_train, Y_train, _, _ = fetch_train()
    X_test, Y_test, _, test_labels  = fetch_test()
  
    # Create the network
    net = NeuralNetwork(layers, lr, batch_size, epochs, rng)

    # Train the network
    net.train(X_train, Y_train)

    # Evaluate the network
    net.evaluate(X_test, test_labels)

if __name__ == "__main__":
    main()