import torch


x= torch.tensor([1.0,2,3], requires_grad= True)
t= torch.tensor([2.0,4,6], requires_grad= True)

W= 3*x+ 2*t
output= W.sum()
print(t.grad)