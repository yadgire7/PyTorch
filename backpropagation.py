# CHAIN RULE
'''
Let's say we have input 'x' and pass it to activation fuction 'a' to get 'y'
and then pass it to another activation function 'b' to get final output 'z'

dz/dx = dy/dx * dz/dy
'''

# COMPUTATIONAL GRAPH
'''
For every operation we do on tensors, pytorch creates a computation graph
'''

# GRADIENT OF LOSS
'''
d(loss)/dx = d(loss)/dz * dz/dx
'''

'''
1. foraward pass (calculate loss)
2. compute local gradients
3. backward pass: (compute d(loss)/s(weights) using chain rule)
'''

import torch

# create a tensor

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# forward pass

y_hat = w*x
loss = (y_hat - y)**2

print(f"loss: {loss}")

# backward pass
loss.backward()
print(f"gradient: {w.grad}")

# update weights
# repeat forward pass and backpropagation until convergence(unchanged weights for consecutive passses)
