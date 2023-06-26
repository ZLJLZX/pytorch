import torch

torch.manual_seed(0)


x = torch.rand(4,5)

print(f'原来元素：{x}')

print(f'某个x[0][0]:{x[0][0]}')

print(f'第一行x[0,:]:{x[0,:]}')

print(f'第一列x[:,1]:{x[:,1]}')



print(f'子矩阵（左闭右开）x[1:3,1:3]:{x[1:3,1:3]}')