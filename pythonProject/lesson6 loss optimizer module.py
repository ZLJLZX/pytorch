import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)
X = torch.tensor([[1,2],[2,3],[4,6],[3,1]],dtype=torch.float32)
y = torch.tensor([[8],[13],[26],[9]],dtype=torch.float32)
# y = 2*x1 + 3 * x2

# w = torch.rand(2,1, requires_grad=True, dtype=torch.float32) 模型内已设定

iter_count = 500
lr = 0.005


# def forward(x):
#     return torch.matmul(x,w)  # 4*1

#用module 替代forward

class MyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        #self.w = torch.nn.Parameter(torch.rand(2,1, dtype= torch.float32)) #自动设置为requires_grad  且加入到自定义模型的参数中
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x ):  #实际中执行的预测函数
        #return torch.matmul(x, self.w)
        return self.linear(x)






#用criterion 替代
# def loss(y, y_pred):
#     return ((y-y_pred)**2/2).sum()#原本返回为向量4*1  .sum()后相加为标量

model = MyModel() #创建模型

criterion = torch.nn.MSELoss(reduction='sum')   #二分类 BCE 多分类 crossentropy
# def gradient(x,y,y_pred):
#     return torch.matmul(x.T, y_pred-y)
# optimizer = torch.optim.SGD([w,],lr)
optimizer = torch.optim.SGD(model.parameters(),lr)  #删去w 因为已经在模型内设置parameter   所有的parameter 会在这里以iterater 形式传入
for i in range(iter_count):
    #三步走
    #第一步  forward pred的值  #第二步  算loss
    #y_pred = forward(X)  # 用model替代
    y_pred = model(X)

    l = criterion(y_pred, y)
    #l = loss(y, y_pred)
    print(f'iter{i}, loss{l}')

    #反向传播
    l.backward()
    optimizer.step()#参数更新
    optimizer.zero_grad()#清理导数 防止积累


    #用optimizer 替代
    # with torch.no_grad():
    #     w -= lr * w.grad
    #     w.grad.zero_()
    #清空梯度
    # #backward 算grad
    # grad = gradient(X, y, y_pred)
    #
    # #更新参数
    # w -= lr * grad


#print(f'final parameter:{model.w}')

print(f'final parameter:{model.linear.weight}')
x1 = 4
x2 = 5

#2*4 + 3* 5 = 23

print(model(torch.tensor([[x1, x2]], dtype = torch.float32)))#  需要输入二维数组