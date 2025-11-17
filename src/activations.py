import numpy as np

# Predefined activations
def get_activation(name):
  if name == 'tanh':
    return np.tanh, lambda x: 1 - np.tanh(x)**2

  elif name == 'sigmoid':
    sig = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    return sig, lambda x: sig(x) * (1 - sig(x))

  elif name == 'relu':
    return lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float)

  elif name == 'softmax':
        def softmax(x):
            # Subtract max for numerical stability
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        def softmax_derivative(x):
            return np.ones_like(x)
        return softmax, softmax_derivative

  elif name is None or name =='linear':
    return None, lambda x: np.ones_like(x)
    
  else:
    raise ValueError(f"Unsupported activation function: {name}")