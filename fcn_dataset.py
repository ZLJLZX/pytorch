import os
import torch.utils.data as data
import numpy as np
import torch
import transforms
from PIL import Image
import matplotlib.pyplot as plt


class VOCSegmentation(data.Dataset):
    def __init__(self, img_path, gt_path, txt_file, train_val='train', base_size=520, crop_size=480, flip_prob=0.5):#原始图像地址 标注图像地址 使用到的图片 训练或验证阶段？ batchsize
        super(VOCSegmentation, self).__init__()

        if train_val == 'train':#如是训练阶段
            with open(txt_file, 'r') as f:
                data = [data.strip() for data in f.readlines() if len(data.strip()) > 0] #读取后变为列表形式

                #目的 统一尺寸 float 归一化 对原图像增强
            self.transforms = transforms.Compose([transforms.RandomResize(int(base_size*0.5), int(base_size*2)),#统一尺寸 需要随机裁剪 进行缩放 防止裁剪到过小区域
                                                  transforms.RandomHorizontalFlip(flip_prob),#随机反转 原图和标注需要同时进行
                                                  transforms.RandomCrop(crop_size),#随机裁剪 裁剪后一致 原图用0 标注为255 填充 255不会考虑在内  原图放左上角
                                                  transforms.ToTensor(),#只对原图像 三件事 转为tensor 灰度0-255到0-1 转化（H,W,C）到（C,H,W）  标注转为tensor 范围0-255不变 0为背景 255边界
                                                  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) #三通道归一化 mean std （）
                                                  ])
        else:
            with open(txt_file, 'r') as f:
                data = [data.strip() for data in f.readlines() if len(data.strip()) > 0]
            self.transforms = transforms.Compose([transforms.RandomResize(base_size, base_size),#520 520
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        self.img_files = [os.path.join(img_path, i + '.jpg') for i in data] #拼接地址为list
        self.gt_files = [os.path.join(gt_path, i + '.png') for i in data]

    def __getitem__(self, index):
        img = Image.open(self.img_files[index]) #利用图像的索引 通过列表 获取原始图像与标注图像
        target = Image.open(self.gt_files[index])
        img, target = self.transforms(img, target) #进行transform
        return img, target  #输出原始和标注图像

    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs






