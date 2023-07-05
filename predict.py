import os
import time
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from fcn_model import fcn_resnet50
from matplotlib import pyplot as plt


def main():
    aux = False  # inference time not need aux_classifier
    classes = 20

    # check files
    weights_path = './save_weights/model.pth' #模型参数
    img_path = 'image.jpg'#指定图片
    assert os.path.exists(weights_path), f"weights {weights_path} not found." #确认上面两个是否存在
    assert os.path.exists(img_path), f"weights {img_path} not found."


    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #指定device

    # create model
    model = fcn_resnet50(aux=aux, num_classes=classes+1)#构建模型对象

    # delete weights about aux_classifier
    weights_dict = torch.load(weights_path, map_location='cpu')['model'] #加载模型参数文件中的model信息（不加载epoch、优化器 学习率调度器）
    for k in list(weights_dict.keys()):
        if "aux" in k:#删去辅助网络
            del weights_dict[k]

    # load weights (  去除辅助网络后的参数)
    model.load_state_dict(weights_dict)
    model.to(device)#送入device

    # preprocess image  预处理
    img_transforms = transforms.Compose([transforms.Resize(520),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))])

    original_img = Image.open(img_path) #打开图像
    img = img_transforms(original_img)#预处理
    img = torch.unsqueeze(img, dim=0)#最前维度加一 （batch size=1）

    model.eval()
    with torch.no_grad():#创建一个值全为0的图像
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time.time()#测试推理速度  前面加了一个0  第二次才开始检测速度
        output = model(img.to(device))
        t_end = time.time()
        print("inference time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)#取出21类中最大的类别 为预测结果 放入
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        #以下为保存图像
        # mask = Image.fromarray(prediction)
        # mask.save("predict_result.png")

    plt.subplot(121)
    plt.imshow(np.array(original_img))
    plt.subplot(122)
    plt.imshow(prediction)
    plt.show()


if __name__ == '__main__':
    main()