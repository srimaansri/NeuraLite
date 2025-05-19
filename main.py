import numpy as np
from models.model import NeuralNetwork
from models.layers import Linear, ReLU, Sigmoid
from models.losses import BinaryCrossEntropy

# Set seed for reproducibility
np.random.seed(42)

# Dummy input: 2 features, 5 samples
x = np.random.randn(2, 5)

# Dummy binary labels: shape (1, 5)
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

# Calculate loss
loss_fn = BinaryCrossEntropy()
loss = loss_fn.forward(y_pred, y_true)
print("\nLoss:", loss)

# Backward pass
grad = loss_fn.backward()
model.backward(grad, learning_rate=0.01)
