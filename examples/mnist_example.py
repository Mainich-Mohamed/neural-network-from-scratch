"""
MNIST Example for Neural Network from Scratch
Demonstrates 96%+ accuracy on handwritten digit classification
"""
import os
import sys
import numpy as np

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from layer import Layer
from network import NeuralNetwork
from utils import load_mnist, to_one_hot, accuracy

def main():
    # Set seed for reproducibility
    np.random.seed(42)
    
    print("Loading MNIST data...")
    (X_train, y_train), (X_test, y_test) = load_mnist()

    # Normalize pixel values to [0, 1]
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Convert labels to one-hot encoding (for cross-entropy loss)
    y_train_onehot = to_one_hot(y_train)
    y_test_onehot = to_one_hot(y_test)

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train_onehot.shape}")

    # Build network: 784 → 128 → 64 → 10
    layers = [
        Layer(128, 784, activation='relu'),
        Layer(64, 128, activation='relu'),
        Layer(10, 64, activation='linear')  # Output layer: logits for cross-entropy
    ]

    net = NeuralNetwork(layers, loss_function="categorical_crossentropy")

    # Trained on subset for faster testing
    train_size = 10000  # Reduce for faster training
    X_train_subset = X_train[:train_size]
    y_train_subset = y_train_onehot[:train_size]

    print("Starting training...")
    net.train(
        X_train_subset, 
        y_train_subset, 
        epochs=50, 
        learning_rate=0.001, 
        batch_size=64
    )

    # Evaluate
    train_preds = net.forward(X_train_subset)
    train_acc = accuracy(train_preds, y_train_subset)
    print(f"\nTraining Accuracy: {train_acc:.4f}")

    test_preds = net.forward(X_test)
    test_acc = accuracy(test_preds, y_test_onehot)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()