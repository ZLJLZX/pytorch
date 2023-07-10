from torchvision import models
import torch
from torchsummary import summary
from resnet import resnet50


# ------------------------------------------------------------
#  任务一 ：
#  1、将模型A 作为backbone，修改为 模型B
#  2、模型A的预训练参数 加载到 模型B上
# ------------------------------------------------------------

resnet_modified = resnet50()
new_weights_dict = resnet_modified.state_dict()

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
weights_dict = resnet.state_dict()

for k in weights_dict.keys(): #迭代读取key
    if k in new_weights_dict.keys() and not k.startswith('fc'): #名字在yolo1 且不是全连接 赋值
        new_weights_dict[k] = weights_dict[k]
resnet_modified.load_state_dict(new_weights_dict)#加载参数到框架内
# resnet_modified.load_state_dict(new_weights_dict, strict=False)


# --------------------------------------------------
#  任务二：
#  冻结与训练好的参数
# --------------------------------------------------
params = []
train_layer = ['layer5', 'conv_end', 'bn_end'] #需要训练的层名称
for name, param in resnet_modified.named_parameters():#迭代参数名和参数
    if any(name.startswith(prefix) for prefix in train_layer):#先判断是否是需要训练的三个层内
        print(name)
        params.append(param)# 是则加入到params 列表内
    else:
        param.requires_grad = False#不是 则冻结 不更新梯度

optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=5e-4)

