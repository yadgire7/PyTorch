import torch

# create a 1D empty tensor
tensor_1d = torch.empty(1)
print(f"empty 1D tensor: {tensor_1d}")

# output: tensor([0.])

# create a 2D empty tensor
tensor_2d = torch.empty(2,3)
print(f"empty 2D tensor: {tensor_2d}")

'''
output:
tensor([[1.9421e+31, 1.1722e-19, 1.3563e-19],

        [1.1815e+22, 1.6246e+19, 2.9972e+32]]
)
'''

# create a 3D empty tensor
tensor_3d = torch.empty(2, 2, 3)
print(f"empty 3D tensor: {tensor_3d}")

'''
tensor([[[0.0000e+00, 0.0000e+00, 0.0000e+00]
,
         [0.0000e+00, 1.4013e-45, 0.0000e+00]
],

        [[1.4013e-45, 0.0000e+00, 1.4013e-45]
,
         [0.0000e+00, 0.0000e+00, 1.9073e-05]
]])
'''

# create a tensor with random values (between 0 and 1)
tensor_random = torch.rand(2,3)
print(f"Random tensor: {tensor_random}")

'''
tensor([[0.1912, 0.0660, 0.5843],
        [0.9997, 0.3077, 0.5795]])
'''

# tensor with all ones

tensor_ones = torch.ones(3,3)
print(f"Tensor with ones(float): {tensor_ones}")
print(tensor_ones.dtype)

'''
Output:
Tensor with ones(float): tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])
torch.float32
'''

# change the datatype from float to int:
tensor_ones_int = torch.ones(3, 3, dtype=torch.int)
print(f"Tensor with ones(int): {tensor_ones_int}")
print(tensor_ones_int.dtype)

'''
output:
Tensor with ones(int): tensor([[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]], dtype=torch.int32)
torch.int32
'''

# transpose a tensor

tr = tensor_random.transpose(0, 1)
print(f"original: {tensor_random.shape}")
print(f"transpose: {tr.shape}")

'''
output:
original: torch.Size([2, 3])
transpose: torch.Size([3, 2])
'''

tr[1][1] = 2
print(tr)

'''
tensor([[0.7462, 0.0028],
        [0.7517, 2.0000],
        [0.2870, 0.6263]])
'''

'''
Note: Both, tensor_random and tr are stored in the sam elocation
i.e it is the same data, just the accessing pattern is different
'''

print(id(tensor_random.storage()) == id(tr.storage()))
# output: True

# convert tensor to numpy array

np_tr = tr.numpy()
print(type(tr))
print(type(np_tr))

# adding some constant to a tensor

add_tensor = tensor_ones.add_(1)
print(add_tensor)

'''
tensor([[2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.]])
'''