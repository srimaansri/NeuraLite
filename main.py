import numpy as np
from models.model import NeuralNetwork
from models.layers import Linear, ReLU, Sigmoid

# Dummy input: 2 features, 5 samples
x = np.random.randn(2, 5)

# Dummy ground truth: 1 output per sample
y_true = np.array([[0, 1, 0, 1, 1]])

# Initialize model
model = NeuralNetwork()
model.add(Linear(2, 4))    # Input -> Hidden
model.add(ReLU())
model.add(Linear(4, 1))    # Hidden -> Output
model.add(Sigmoid())       # Output activation

# Forward pass
y_pred = model.forward(x)
print("Forward output (y_pred):")
print(y_pred)

# Backward pass with dummy gradient (pretend we used loss)
# We’ll fake a loss gradient for now until we write real loss
grad_output = y_pred - y_true  # Gradient of MSE or BCE-like loss
model.backward(grad_output, learning_rate=0.01)
