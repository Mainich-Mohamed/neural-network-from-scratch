"""
XOR Example for Neural Network from Scratch
Demonstrates solving the classic non-linear XOR problem
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from layer import Layer
from network import NeuralNetwork

def main():
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    layers = [
        Layer(8, 2, activation='tanh'),
        Layer(1, 8, activation='sigmoid')
    ]
    net = NeuralNetwork(layers, loss_function="binary_crossentropy")

    net.train(X, y, epochs=500, learning_rate=0.01, batch_size=4)

    print("\nPredictions:")
    preds = net.forward(X)
    for i in range(len(X)):
        print(f"{X[i]} → {preds[i][0]:.4f} (target: {y[i][0]})")

if __name__ == "__main__":
    main()