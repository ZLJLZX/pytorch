import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader



# class MyDataset(Dataset):
#     def __init__(self):
#         pass
#     def __getitem__(self, item):
#         pass
#
#     def __len__(self):
#         pass


class MyDataset(Dataset):
    def __init__(self):#载入数据
        txt_data = np.loadtxt('./data.txt', delimiter=',')
        self._x = txt_data[:,:2]
        self._y = txt_data[:,2]
        self._len = len(txt_data)


    def __getitem__(self, item):#返回相应index的数据

        return self._x[item], self._y[item]

    def __len__(self):#返回整个数据集的长度

        return self._len



data = MyDataset()

print(len(data))


first = next(iter(data))
print(first)
print(type(first[0]), type(first[1]))

class MyDataset1(Dataset):
    def __init__(self):#载入数据
        txt_data = np.loadtxt('./data.txt', delimiter=',')
        self._x = torch.from_numpy(txt_data[:,:2])
        self._y = torch.from_numpy(txt_data[:,2])
        self._len = len(txt_data)


    def __getitem__(self, item):#返回相应index的数据

        return self._x[item], self._y[item]

    def __len__(self):#返回整个数据集的长度

        return self._len


data1 = MyDataset1()

print(len(data1))


first = next(iter(data1))
print(first)
print(type(first[0]), type(first[1]))



dataloader = DataLoader(data1, batch_size = 1, shuffle = True, drop_last=True, num_workers=0)

a = 0
for data_val, label_val in dataloader:
    print('x:',data_val,'y:',label_val)
    a += 1
print('iteration:', a)