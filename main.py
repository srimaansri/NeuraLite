import numpy as np
from models.layers import Linear, ReLU

# Create dummy input (e.g., 3 features, 5 samples)
x = np.random.randn(3, 5)

# Initialize layers
fc = Linear(in_features=3, out_features=2)
activation = ReLU()

# Forward pass
out = fc.forward(x)
activated = activation.forward(out)

print("Forward output:")
print(activated)

# Fake gradient coming from next layer (same shape as output)
grad_from_next = np.random.randn(2, 5)

# Backward pass
grad_relu = activation.backward(grad_from_next)
grad_input = fc.backward(grad_relu, learning_rate=0.01)

print("\nBackward output (gradient w.r.t input):")
print(grad_input)
