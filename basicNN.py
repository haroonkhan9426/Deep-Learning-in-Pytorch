import torch
import numpy as np

def activation(x):
    # Sigmoid function
    print('Value of x: ', x)
    return 1/(1 + torch.exp(-x))

def neuralNetwork(f, w, b):
    # Solution using torch.sum
    return activation(torch.sum(f * w) + b)
    # Solution using matmul
    # return activation(torch.matmul(f, w) + b)

# Set seed to get same sudo random numbers
torch.manual_seed(5)
# Features = input x values
# Weights = Weights (W1, W2, .....Wn)
# Bias = Single bias value
features = torch.randn((1, 5))
weights = torch.randn_like(features)
bias = torch.randn((1,1))

print(features.shape)
print(weights.shape)
# output = neuralNetwork(features, weights.view(5, 1), bias)
output = neuralNetwork(features, weights, bias)
print(output)










    