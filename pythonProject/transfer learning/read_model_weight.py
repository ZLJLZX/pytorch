import torch


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Linear(4, 3),
        )
        self.layer2 = torch.nn.Linear(3, 6)

        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(6, 7),
            torch.nn.Linear(7, 5),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


net = MyModel()
print(net)

# -----------------------------------------------------------
# net.modules()、每层递归答应   net.named_modules() 在前面的基础上 多了名称
# -----------------------------------------------------------
# for layer in net.modules():
#     #print(type(layer)) 类别
#     print(layer)#内容 输入输出特征数
#     #break

# <class '__main__.MyModel'>   整个网络类型
# <class 'torch.nn.modules.container.Sequential'> 第一层类型
# <class 'torch.nn.modules.linear.Linear'>第一层内 第一个全连接类型
# <class 'torch.nn.modules.linear.Linear'>第一层 第二个
# <class 'torch.nn.modules.linear.Linear'>
# <class 'torch.nn.modules.container.Sequential'>
# <class 'torch.nn.modules.linear.Linear'>
# <class 'torch.nn.modules.linear.Linear'>








#
# for name, layer in net.named_modules():
#     #print(name, type(layer))
#     print(name, layer)

# -----------------------------------------------------------
# net.children()、 只打印子网络  各层内的子网络  net.named_children()  多了子网络名称
# -----------------------------------------------------------

# for layer in net.children():
#     print(layer)

# for name, layer in net.named_children():
#     print(name, layer)


# -----------------------------------------------------------
# net.parameters()、  每层参数  net.named_parameters() 层名 加参数
# -----------------------------------------------------------

# for param in net.parameters():
#     print(param.shape)


# for name, param in net.named_parameters():
#     print(name, param.shape)

# -----------------------------------------------------------
# net.state_dict()   网络中 参数名与参数 以字典方式迭代
# -----------------------------------------------------------

for key, value in net.state_dict().items():
    print(key, value.shape)



