import numpy as np

class NeuralNetwork:
  def __init__(self, layers, loss_function=None):
    self.layers = layers
    self.loss_function = loss_function

  def forward(self, inputs):
    """
        Forward pass through all layers.

        Args:
            inputs: Input data (batch_size, num_inputs)

        Returns:
            Final output after passing through all layers
        """

    for layer in self.layers:
      inputs = layer.forward(inputs)
    return inputs

  def loss(self, y_pred, y_true):
    if self.loss_function is None:
      raise ValueError("No loss function provided")

    loss = self.loss_function(y_pred, y_true)
    return loss

  def compute_loss_and_grad(self, y_pred, y_true):
    """Returns (loss_value, grad_wrt_y_pred)"""
    if self.loss_function == "mse":
      loss = np.mean((y_pred - y_true) ** 2)
      grad = 2 * (y_pred - y_true) / y_pred.size # dL/dy_pred
      return loss, grad

    elif self.loss_function == "binary_crossentropy":
        # Clip predictions to prevent log(0) and division by zero
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -np.mean(
            y_true * np.log(y_pred_clipped) + 
            (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        
        grad = (
            -(y_true / y_pred_clipped) + 
            (1 - y_true) / (1 - y_pred_clipped)
        ) / y_pred.shape[0]  # Divide by batch size for mean
        return loss, grad

    elif self.loss_function == "categorical_crossentropy":
      # Compute softmax probabilities
      exp_logits = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
      softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
      
      # Clip probabilities to prevent log(0)
      softmax_probs = np.clip(softmax_probs, 1e-7, 1 - 1e-7)
      
      # Compute loss
      loss = -np.mean(np.sum(y_true * np.log(softmax_probs), axis=1))
      
      # Gradient for cross-entropy with softmax: dL/dz = softmax(z) - y_true
      grad = (softmax_probs - y_true) / y_true.shape[0]
      
      return loss, grad

    else:
      raise ValueError(f"Unsupported loss function: {self.loss_function}")

  def backward(self, y_pred, y_true):
    """Compute gradients for all layers"""
    # Get gradient from loss
    loss, grad_output = self.compute_loss_and_grad(y_pred, y_true)

    # Step 2: Backpropagate through layers in REVERSE order
    for layer in reversed(self.layers):
      grad_output = layer.backward(grad_output)

    return loss

  def update_params_adam(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Update all layers using Adam optimizer"""
    for layer in self.layers:
      layer.update_params_adam(learning_rate, beta1, beta2, epsilon)

  def train_step(self, X, y, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Single training step with Adam"""
    y_pred = self.forward(X)
    loss = self.backward(y_pred, y)
    self.update_params_adam(learning_rate, beta1, beta2, epsilon)
    return loss

  def train(self, X, y, epochs=100, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=None):
    """
    Train the network using Adam optimizer.
    
    Args:
        X: Input data (n_samples, n_features)
        y: Target labels (n_samples, n_outputs)
        epochs: Number of passes through the dataset
        learning_rate: Adam learning rate (default 0.001)
        batch_size: If None, use full batch
    """
    n_samples = X.shape[0]
    batch_size = batch_size or n_samples

    for epoch in range(epochs):
      epoch_loss = 0.0
      # Shuffle data for each epoch
      indices = np.random.permutation(n_samples)
      X_shuffled = X[indices]
      y_shuffled = y[indices]

      for i in range(0, n_samples, batch_size):
        batch_X = X_shuffled[i:i+batch_size]
        batch_y = y_shuffled[i:i+batch_size]

        loss = self.train_step(batch_X, batch_y, learning_rate, beta1, beta2, epsilon)
        epoch_loss += loss * len(batch_X)  # Weight by batch size

      avg_loss = epoch_loss / n_samples
      if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")  