from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits