import torch
from load import test

# Create example tensors
a = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
b = torch.tensor([5.0, 6.0, 7.0, 8.0], device='cuda')

# Perform multiply-add operation
c = test(a, b)

print(c)