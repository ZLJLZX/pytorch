import torch.nn as nn
import torch
import matplotlib.pyplot as plt


class Self_Attention(nn.Module):
    def __init__(self, dim, dk, dv):
        super(Self_Attention, self).__init__()
        self.scale = dk ** -0.5
        #q k输出要一致 v不需要
        self.q = nn.Linear(dim, dk)
        self.k = nn.Linear(dim, dk)
        self.v = nn.Linear(dim, dv)


    def forward(self, x):
        q = self.q(x)  #q [1,4,2]
        k = self.k(x)  #k [1,4,2]
        v = self.v(x)  #v [1,4,3]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)#最后一个维度 做softmax

        x = attn @ v
        return x


att = Self_Attention(dim=2, dk=2, dv=3)
x = torch.rand((1, 4, 2))
output = att(x)
print(output)




#inputs embeding size = positional embedding size(维度相同 需要相加)