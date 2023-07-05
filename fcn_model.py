import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from collections import OrderedDict
from backbone import resnet50


class IntermediateLayerGetter(nn.ModuleDict):

    def __init__(self, model, return_layers): # return_layers {'layer4:'out','layer3':'aux'}
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):  #先检查是否在名字里
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers #先保存一份字典（后续会移除元素 导致无法使用）
        return_layers = {str(k): str(v) for k, v in return_layers.items()} # 重置加验证字典格式

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()#定义有序字典
        for name, module in model.named_children():#迭代出每一层的名字和模块结构
            layers[name] = module #并装入layers这个字典
            if name in return_layers:
                del return_layers[name]
            if not return_layers:#当字典为空后 break出for循环   最后两层不会往里面装  即整个backbone去除最后两个层
                break

        super(IntermediateLayerGetter, self).__init__(layers)#把layers 换成moduledict的格式
        self.return_layers = orig_return_layers#一开始备份的字典给self

    def forward(self, x):
        out = OrderedDict()#先新建有序字典 让输入一层一层的过
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name] #直到第三个大的block
                out[out_name] = x#记录第三第四层的输出
        return out#aux【4,1024,60,60】； out【4,2048,60,60】


class FCN(nn.Module):#传入backbone 主副最后层
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, aux_classifier=None):
        super(FCN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier#主
        self.aux_classifier = aux_classifier#副

    def forward(self, x):# x为四维tensor
        input_shape = x.shape[-2:]#记录后两位（高和宽  用于最后上采样到和原图尺寸一样
        features = self.backbone(x)#x传给backbone 得到features

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)  #流入主classifier 双线性差值上采样
        result["out"] = x #给到out字典的值

        if self.aux_classifier is not None: #是否辅助支路
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x
        return result #一个字典 含主副支路的值


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False), #3*3的卷积
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)#跟1*1的卷积层
        ]

        super(FCNHead, self).__init__(*layers)


def fcn_resnet50(aux, num_classes=21):
    backbone = resnet50(replace_conv=[False, True, True]) #继承res50 backbone

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)#除去最后两层的backbone

    aux_classifier = None #判断是否需要做辅助模块
    if aux:
        aux_classifier = FCNHead(1024, num_classes)#需要则搭建旁支
    classifier = FCNHead(2048, num_classes)
    model = FCN(backbone, classifier, aux_classifier)
    return model


if __name__ == '__main__':
    model = fcn_resnet50(True, num_classes=21)
    print(model)
    # x = torch.rand((1, 3, 224, 224))
    # predict = model(x)
    # print(predict)