import numpy as np

class Neuron:
    def __init__(self, weights, bias, activation=None):
        """
          Initialize a neuron with weights and bias

          Args:
              weights: Weight vector for the neuron
              bias: Bias value for the neuron
              activation: Activation function for the neuron
        """
        self.weights = weights
        self.bias = bias
        self.activation = activation


    def compute(self, inputs):
      """
        Compute neuron output for given inputs

        Args:
            inputs: Input vector

        Returns:
            Neuron output after applying activation function
      """
      # Vectorized weighted sum
      z = np.dot(inputs, self.weights) + self.bias

      # Apply activation function if provided
      if self.activation is not None:
          z = self.activation(z)
      return z