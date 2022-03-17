from torchviz import make_dot
from NN import NeuralNetwork
import torch

try:
    model = torch.load("linear1")
    model.eval
except:
    print("Creating new model")
    model = NeuralNetwork().to(device)
    model.double()

x = torch.randn(10, dtype=torch.double)
y = model(x)

make_dot(y, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")