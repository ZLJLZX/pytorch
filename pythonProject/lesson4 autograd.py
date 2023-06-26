# z = 2* a + a*b
import torch

a = torch.tensor(3., requires_grad=True)#跟踪计算过程 计算tensor的导数
b = torch.tensor(4., requires_grad= True)


print(a.requires_grad)



a = torch.tensor(3.)

a.requires_grad_(True)
print(a.requires_grad)


f1 = 2 * a
f2 = a * b
z = f1 + f2
print(f1.grad_fn, f2.grad_fn)
print(z)#计算导数方法为加法


z.backward()# 标量可以直接backward vector 需要 传进去另外一个vector  然后雅可比矩阵相乘 得到一个结果
print(f'a.grad = {a.grad}')
print(f'b.grad = {b.grad}')


#不需要梯度更新  两种方法 从计算图中剥离 分离 
with torch.no_grad():
    f3 = a * b
    print(f'f2.requires_grad = {f2.requires_grad}')
    print(f'f3.requires_grad = {f3.requires_grad}')


a1 = a.detach()
print(f'a.requires_grad = {a.requires_grad}')
print(f'a1.requires_grad = {a1.requires_grad}')