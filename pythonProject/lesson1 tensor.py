import numpy as np
import torch

arr = [[3,1],[2,1]]
x = torch.tensor(arr, dtype=torch.float32)
print(x)

x1 = torch.rand(3,2)
print(x1)

x = torch.zeros(4,5)
print(x)

x= torch.rand(2,3)



y= torch.zeros_like(x)
z = torch.ones_like(x)
print(y)

arr_np = np.random.rand(3,4)
x = torch.from_numpy(arr_np)

print(f'numpy arr:{arr_np}')
print(f'tensor:{x}')


arr_np[0,0]=777
x[0,1] =-100

print(f'change numpy:{arr_np}, tensor:{x }')



y= x.numpy()
print(y)