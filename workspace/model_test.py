import torch
import torch.nn as nn

linear1 = nn.Linear(10, 20)
input = torch.randn(30, 10)
output = linear1(input)
print(output.shape)