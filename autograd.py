'''
Learning the autograd module from pytorch
- gradient descent
'''

import torch

# create a tensor with gradient param
x = torch.randn(3, requires_grad=True)
print(x)
'''
tensor([ 1.0973,  0.0092, -0.7855], requires_grad=True)
'''

y = x+2
print(y)
'''
tensor([0.2156, 2.5393, 2.5447], grad_fn=<Add
Backward0>)
'''

z = y*y*2
print(z)
'''
tensor([18.7682,  3.2792, 11.2734], grad_fn=<
MulBackward0>)
'''

m = z.mean()
print(m)
'''
tensor(8.8082, grad_fn=<MeanBackward0>)
'''

print(m.backward())
print(x.grad)
'''
None
tensor([2.0240, 2.4456, 3.7093])
'''