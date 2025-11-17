# Neural Network from Scratch 🧠

![Deep Learning Diagram]([https://lamarr-institute.org/wp-content/uploads/deepLearn_2_EN.png](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41377-024-01590-3/MediaObjects/41377_2024_1590_Fig3_HTML.png))

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
