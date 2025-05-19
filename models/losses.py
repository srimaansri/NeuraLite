import numpy as np

class BinaryCrossEntropy:
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def forward(self, y_pred, y_true):
        self.y_true = y_true
        self.y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Prevent log(0)
        loss = -np.mean(y_true * np.log(self.y_pred) + (1 - y_true) * np.log(1 - self.y_pred))
        return loss

    def backward(self):
        # Derivative of BCE with respect to predictions
        return (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred) * self.y_true.shape[1])
