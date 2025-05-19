class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)
        return grad_output

    def predict(self, x):
        output = self.forward(x)
        return output > 0.5  # for binary tasks

    def summary(self):
        print("Model architecture:")
        for layer in self.layers:
            print(f" - {layer.__class__.__name__}")
