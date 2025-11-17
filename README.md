<<<<<<< HEAD
# neural-network-from-scratch
A fully functional neural network implementation from scratch using only NumPy. Features Adam optimizer, backpropagation, and achieves 96% accuracy on MNIST.
=======
# Neural Network from Scratch 🧠

A fully functional neural network implementation built entirely from NumPy, featuring:
- ✅ Forward and backward propagation
- ✅ Adam optimizer with bias correction
- ✅ Multiple activation functions (ReLU, tanh, sigmoid, softmax)
- ✅ Proper weight initialization (Xavier/He)
- ✅ Numerical stability (clipping, softmax stabilization)
- ✅ Batch training with shuffling

## Results

### XOR Problem
- **100% accuracy** - solved the classic non-linear problem

### MNIST Handwritten Digits
- **96.07% test accuracy** with just 10k training samples
- Architecture: `784 → 128 → 64 → 10`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### XOR Problem Example
```bash
python examples/xor_example.py
```

### MNIST Classification Example
```bash
python examples/mnist_example.py
```
>>>>>>> master
