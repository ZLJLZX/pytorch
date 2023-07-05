import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class Self_Attenion(nn.Module):
      def __init__(self, dim ,dk, dv):
          super(Self_Attenion, self).__init__()
          self.scale = dk ** -0.5
          self.q = nn.Linear(dim,dk)
          self.k = nn.Linear(dim,dk)
          self.v = nn.Linear(dim,dv)

      def forward(self, x):
          q = self.q(x)
          k = self.k(x)
          v = self.v(x)

          attention = (q @ k.transpose(-2,-1)) * self.scale
          attention = attention.softmax(dim=-1)

          x = attention @ v
          return x

attn = Self_Attenion(dim = 2, dk=2, dv= 3)

x = torch.rand((1,4,2))
output = attn(x)
print(output)





def postitional_encoding(encoding_dim, max_len=1000):
    pe = torch.zeros(max_len, encoding_dim)
    v = torch.arange(max_len,dtype=torch.float32).reshape(-1,1)/torch.pow(10000,torch.arange(0,encoding_dim,2,dtype=torch.float32)/encoding_dim)
    pe[:, 0::2] = torch.sin(v)
    pe[:,1::2]=torch.cos(v)
    return pe

num_token, encoding_dim = 60,32
pe = postitional_encoding(encoding_dim)
plt.plot(torch.arange(num_token), pe[:60, 5:8])
plt.xlabel('num token')
plt.ylabel('PE value')
plt.show()