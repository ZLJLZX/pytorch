import torch

torch.manual_seed(0)


# view reshape 用的相同数据地址 不改原数据内容
x = torch.rand(20)#20*1
print(f'x:{x}, x data ptr:{x.data_ptr()}')
y = x.view(4,5)#4*5
print(f'x.view y:{y}, data ptr:{x.data_ptr()}')


y = x.reshape(4,5)
print(f'x.reshape:{y}, data ptr:{x.data_ptr()}')


xt = y.T # 5*4
#  z = xt.view(1,20)  会报错 RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
#需要先加continuouse



#分配新内存空间 与原数据不同地址
z = xt.contiguous().view(1,20)# 兼容input
print(f'z:{z}，data ptr:{z.data_ptr()}')


z = xt.reshape(1,20)
print(f'z:{z}，data ptr:{z.data_ptr()}')




y = x.unsqueeze(0)
print(f'after unsequeeze 000 :{y}')#[20]  --- [1,20]
print(f'after unsequeeze shape:{y.shape}')


y = x.unsqueeze(1)
print(f'after unsequeeze 111 :{y}')#[20]  --- [1,20]
print(f'after unsequeeze shape:{y.shape}')



z = y.squeeze(1)
print(z)

z = y.squeeze(0)
print(z)