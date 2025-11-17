import numpy as np
from activations import get_activation

class Layer:
  def __init__(self, num_neurons, num_inputs, activation=None):
      """
        Initialize a layer with multiple neurons

        Args:
            num_neurons: Number of neurons in the layer
            num_inputs: Number of inputs to each neuron
            activation: Activation function for all neurons in the layer
        """
      self.num_neurons = num_neurons
      self.num_inputs = num_inputs

      # Handle string or (func, deriv) tuple
      if isinstance(activation, str):
        self.activation, self.activation_derivative = get_activation(activation)
      elif isinstance(activation, tuple):
        self.activation, self.activation_derivative = activation
      elif activation is None:
        self.activation, self.activation_derivative = get_activation(activation)
      else:
        raise ValueError("Activation function must be a string, tuple (func, deriv), or None")

      # Initialize weights and biases for all neurons
      if self.activation == np.tanh or 'tanh' in str(self.activation):
          std = np.sqrt(1 / self.num_inputs)  # Xavier
      else:  # ReLU or sigmoid
          std = np.sqrt(2 / self.num_inputs)  # He
      self.weights = np.random.randn(num_neurons, num_inputs) * std
      self.biases = np.zeros(num_neurons)  # Biases to zero is standard

      # For backward pass
      self.input = None
      self.z = None

      # Adam optimizer state (initialized to zero)
      self.m_weights = np.zeros_like(self.weights)  # 1st moment (mean)
      self.v_weights = np.zeros_like(self.weights)  # 2nd moment (variance)
      self.m_biases = np.zeros_like(self.biases)
      self.v_biases = np.zeros_like(self.biases)
      self.t = 0  # Timestep counter (for bias correction)

  def forward(self, inputs):
      """
        Compute the output of all neurons in the layer

        Args:
            inputs: Input data (batch_size, num_inputs)

        Returns:
            Layer output (batch_size, num_neurons)
        """
      self.input = inputs

      # Vectorized computation for all neurons
      self.z = np.dot(inputs, self.weights.T) + self.biases

      # Apply activation function if provided
      if self.activation is not None:
         self.output = self.activation(self.z)
      else:
        self.output = self.z
      return self.output

  def backward(self, grad_output):
    """
    Compute gradients w.r.t. weights, biases, and input.

    Args:
        grad_output: Gradient of loss w.r.t. this layer's OUTPUT (dL/dy)

    Returns:
        grad_input: Gradient of loss w.r.t. this layer's INPUT (dL/dx)
    """
    # Activation derivative: dL/dz = dL/dy * dy/dz
    grad_z = grad_output * self.activation_derivative(self.z)

    # Compute gradients for parameters
    # dL/dW = (dL/dz)^T @ input  → shape: (num_neurons, num_inputs)
    batch_size = self.input.shape[0]
    self.grad_weights = np.dot(grad_z.T, self.input) / batch_size
    self.grad_biases = np.mean(grad_z, axis=0)

    # Gradient w.r.t. input for previous layer: dL/dx = dL/dz @ W
    grad_input = np.dot(grad_z, self.weights)

    return grad_input

  def update_params_adam(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam optimizer.
    """
    self.t += 1  # Increment timestep
    
    # Update biased first moment estimate (momentum)
    self.m_weights = beta1 * self.m_weights + (1 - beta1) * self.grad_weights
    self.m_biases = beta1 * self.m_biases + (1 - beta1) * self.grad_biases
    
    # Update biased second raw moment estimate (velocity)
    self.v_weights = beta2 * self.v_weights + (1 - beta2) * (self.grad_weights ** 2)
    self.v_biases = beta2 * self.v_biases + (1 - beta2) * (self.grad_biases ** 2)
    
    # Compute bias-corrected estimates
    m_weights_corr = self.m_weights / (1 - beta1 ** self.t)
    m_biases_corr = self.m_biases / (1 - beta1 ** self.t)
    v_weights_corr = self.v_weights / (1 - beta2 ** self.t)
    v_biases_corr = self.v_biases / (1 - beta2 ** self.t)
    
    # Update parameters
    self.weights -= learning_rate * m_weights_corr / (np.sqrt(v_weights_corr) + epsilon)
    self.biases -= learning_rate * m_biases_corr / (np.sqrt(v_biases_corr) + epsilon)