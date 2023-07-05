import torch
a = torch.LongTensor([[1,2,3,0],[4,5,6,0]])
a = a.data.eq(0)
print(a)

b = a.unsqueeze(1)
c = a.unsqueeze(0)

print(b)
print(c)