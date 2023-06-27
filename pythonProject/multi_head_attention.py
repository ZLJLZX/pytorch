from math import sqrt
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, dim_in, d_model, num_heads=3):
        super(MultiHeadSelfAttention, self).__init__()

        self.dim_in = dim_in   # dim_in = 2
        self.d_model = d_model # dim_d = 6   q k v 总长度为6
        self.num_heads = num_heads # dim_in = 2  每个head 维度6/3 为2

        # 维度必须能被num_head 整除
        assert d_model % num_heads == 0, "d_model must be multiple of num_heads"

        # 定义线性变换矩阵   全连接
        self.linear_q = nn.Linear(dim_in, d_model)
        self.linear_k = nn.Linear(dim_in, d_model)
        self.linear_v = nn.Linear(dim_in, d_model)
        self.scale = 1 / sqrt(d_model // num_heads)

        # 最后的线性层
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.d_model // nh  # dim_k of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)   从【1，4，6】 到 【1，4，3，2】  三头 俩特征  交换 3，4 变【1，3，4，2】 把head维度放到1维  最后 4，2 为 qkv维度 方便并行
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)

        dist = torch.matmul(q, k.transpose(2, 3)) * self.scale  # batch, nh, n, n  得到相似度分数
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n 最后一个维度做softmax

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.d_model)  # batch, n, dim_v  重新变为【1，4，3，2】 三头 俩特征 并变为【1，4，6】

        # 最后通过一个线性层进行变换
        output = self.fc(att)

        return output


x = torch.rand((1, 4, 2))
multi_head_att = MultiHeadSelfAttention(x.shape[2], 6, 3)  # (6, 3)
output = multi_head_att(x)

print(output)