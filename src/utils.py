import numpy as np
import gzip
import os
from urllib.request import urlretrieve

def to_one_hot(y, num_classes=10):
    """Convert labels to one-hot encoding"""
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

def accuracy(y_pred, y_true):
    """Compute accuracy for classification"""
    if y_pred.shape[1] > 1:  # Multi-class
        pred_classes = np.argmax(y_pred, axis=1)
        true_classes = np.argmax(y_true, axis=1)
    else:  # Binary
        pred_classes = (y_pred > 0.5).astype(int).flatten()
        true_classes = y_true.flatten()
    return np.mean(pred_classes == true_classes)

def load_mnist():
    """Load MNIST dataset without TensorFlow"""
    url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Download files if not present
    for key, filename in files.items():
        filepath = f'data/{filename}'
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urlretrieve(url + filename, filepath)
    
    def read_images(filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            return data.reshape(-1, 28*28).astype(np.float32) / 255.0
    
    def read_labels(filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
            return data
    
    X_train = read_images('data/train-images-idx3-ubyte.gz')
    y_train = read_labels('data/train-labels-idx1-ubyte.gz')
    X_test = read_images('data/t10k-images-idx3-ubyte.gz')
    y_test = read_labels('data/t10k-labels-idx1-ubyte.gz')
    
    return (X_train, y_train), (X_test, y_test)