import torch
import numpy as np

scalar = torch.tensor(7) # Scalar value

print(scalar.item())
print(scalar.shape)
print(scalar.ndim)

vector = torch.tensor([1, 2, 3])

print(vector.shape)
print(vector.ndim)

matrix = torch.tensor([[1,3],[3,1]])

print(matrix.shape)
print(matrix.ndim)

tensor = torch.tensor([
    [[1, 2, 3],
    [3, 6, 9],
    [2, 4, 5],
    [4, 2, 1]],

    [[2, 3, 4],
     [5, 2, 5],
     [7, 6, 3],
     [3, 4, 5]]
    ])

print(tensor.shape)
print(tensor.ndim)
