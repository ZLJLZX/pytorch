import numpy as np
np.random.seed(0)


X = np.array([[1,2],[2,3],[4,6],[3,1]],dtype=np.float32)
y = np.array([[8],[13],[26],[9]],dtype=np.float32)
# y = 2*x1 + 3 * x2

w = np.random.rand(2,1)

iter_count = 20
lr = 0.02

def forward(x):
    return np.matmul(x,w)  # 4*1


def loss(y, y_pred):
    return ((y-y_pred)**2/2).sum()#原本返回为向量4*1  .sum()后相加为标量


def gradient(x,y,y_pred):
    return np.matmul(x.T, y_pred-y)



for i in range(iter_count):
    #三步走
    #第一步  forward pred的值  #第二步  算loss
    y_pred = forward(X)
    l = loss(y,y_pred)
    print(f'iter{i}, loss{l}')

    #backward 算grad
    grad = gradient(X, y, y_pred)

    #更新参数
    w -= lr * grad


print(f'final parameter:{w}')


x1 = 4
x2 = 5

#2*4 + 3* 5 = 23

print(forward(np.array([[x1, x2]], dtype = np.float32)))#  需要输入二维数组