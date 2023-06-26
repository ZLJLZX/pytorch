import torch

x = torch.tensor([[1,1],[2,2],[3,3]])
y = torch.tensor([[1,2],[3,4],[5,6]])

print(f'method1: {x+y}')  # -    *  /


print(f'method2: {torch.add(x,y)}')  #add--sub   torch.sub(x,y)   torch.mul(x,y)  torch.div(x,y)



x.add_(y)#x.sub_(y)   x.mul_(y)  x.div_(y)
print(f'method3:{x}')


y = torch.tensor([[1,2],[3,4],[5,6]])

x = torch.tensor([[1,1],[2,2],[3,3]])
print(f'所有元素和：{y.sum()}')  #mean


print(f'每行的和：{y.sum(axis=1)}')#axis = 运算后会改变的轴

print(f'每列的和：{y.sum(axis = 0)}')



print(f'y.T:{y.T}')
print(f'torch.matmul:{torch.matmul(y.T,x)}')
