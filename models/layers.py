import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(out_features, in_features) * 0.01
        self.bias = np.zeros((out_features, 1))
        self.input = None

    def forward(self, x):
        self.input = x
        return np.dot(self.weights, x) + self.bias

    def backward(self, grad_output, learning_rate):
        m = self.input.shape[1]  # batch size
        grad_weights = (1 / m) * np.dot(grad_output, self.input.T)
        grad_bias = (1 / m) * np.sum(grad_output, axis=1, keepdims=True)
        grad_input = np.dot(self.weights.T, grad_output)

        # Gradient descent update
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_input


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output, learning_rate=None):
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input


class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output, learning_rate=None):
        return grad_output * self.output * (1 - self.output)
