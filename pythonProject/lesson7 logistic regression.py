import torch
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(0)
torch.manual_seed(0)


#读取数据


data = datasets.load_breast_cancer()
# print(data.data.shape)
# print(data.target[:50])

X, y = data.data.astype(np.float32), data.target.astype(np.float32)

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X, y, test_size=0.3)# test size 大于0 则为个数  小于0为比例


#特征归一化
sc = StandardScaler()  #分布为0，1 正太分布

X_train_np = sc.fit_transform(X_train_np)  #获取期望和标准差
X_test_np = sc.transform(X_test_np) #用训练组的期望和标准差 处理测试组


X_train = torch.from_numpy(X_train_np)
y_train = torch.from_numpy(y_train_np)

X_test = torch.from_numpy(X_test_np)
y_test = torch.from_numpy(y_test_np)



#构造模型

class MyLogisticRegression(torch.nn.Module):
    def __init__(self, input_feat):
        super().__init__()
        self.linear = torch.nn.Linear(input_feat, 1)
    def forward(self,x): #sigmoid 不需要定义参数 放forward 就好

        y = self.linear(x)
        return torch.sigmoid(y)

input_feat = 30

model = MyLogisticRegression(30)



#Loss and Optimizer

lr = 0.2
num_epochs = 1000
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

#训练
   #迭代  forward 计算loss



for epoch in range(num_epochs):
    # forward 计算loss
    y_pred = model(X_train.view(-1, input_feat))

    loss = criterion(y_pred.view(-1, 1), y_train.view(-1, 1))  #转为[x,1] 自动补齐x的维度 在pytorch中view函数的作用为重构张量的维度，相当于numpy中resize（）的功能

    # backward 更新parameter
    loss.backward()#计算生产gradient
    optimizer.step()#更新参数
    optimizer.zero_grad()# 清理gradient


    with torch.no_grad():
        y_pred_test = model(X_test.view(-1,input_feat))
        y_pred_test = y_pred_test.round().squeeze()#四舍五入 且里面的一列去掉
        total_correct = y_pred_test.eq(y_test).sum()#与另外一个tensor 是否相等
        prec = total_correct.item()/len(y_pred_test)  #里面具体的数 div 总长度
        print(f'epoch:{epoch}, loss:{loss.item()}, precision:{prec}')