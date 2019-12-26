import torch
import numpy as np

def activation(x):
    return 1/(1 + torch.exp(-x))

def neuralNetwork(f, w, b):
    h = torch.matmul(f, w)
    output = activation(h)
    return output
# Set seed to get same sudo random numbers
torch.manual_seed(5)

# Features = input x values
# Weights = Weights (W1, W2, .....Wn)
# Bias = Single bias value
features = torch.randn((1, 5))
weights = torch.randn((5, 1))
bias = torch.randn((1,1))
output = neuralNetwork(features, weights, bias)
print(output)










    