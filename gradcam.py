# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:53

@author: mick.yi

入口类

"""
import re
import os
import numpy as np
import torch
from torch import nn
from torchvision import models
import argparse
from PIL import Image
import cv2
from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from interpretability.guided_back_propagation import GuidedBackPropagation
from model import DANN,DAAN,DDC,DIS,RN
from torchvision import transforms
from collections import OrderedDict
from tqdm import tqdm
def get_net(net_name, weight_path,num_class):
    """
    根据网络名称获取模型
    :param net_name: 网络名称
    :param weight_path: 与训练权重路径
    :return:
    """
    if net_name=="DIS":
        model = DIS.StudentNet(num_class=num_class, base_net='ResNet50')

    pretrained_dict=torch.load(weight_path,map_location='cpu')
    new_state_dict = OrderedDict()
    for i,(k, v) in enumerate(pretrained_dict.items()):
        if "module." in k:
            k=k.replace("module.","")
        new_state_dict[k]=v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()
    return model


def look_all_conv_name(net):
    """
    遍历所有卷积寻找网络的最后一个卷积层的名字（你需要的卷积层的名字）
    :param net:
    :return:
    """
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            print(name)
def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name



def prepare_input(image):
    image = image.copy()

    # 归一化
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = image[np.newaxis, ...]  # 增加batch维

    return torch.tensor(image, requires_grad=True)


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * (1-mask)), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb



    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), norm_image(heatmap)


def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, input_image_name, network, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        cv2.imwrite(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)



def main(args):
    with open(args.image_list,"r") as f:
        image_list=f.readlines()
        f.close()
    net = get_net(args.network,args.weight_path,args.num_class)
        # # Grad-CAM
        # # *** 第二处自定义的地方:选取last convolutional layer name
        # # *** 注释掉后面的代码利用look_last_conv_name(net)查看所有卷积层
        # look_all_conv_name(net)
    if args.network=="DIS":
        layer_name = "base_network.layer4.2.conv3"
    else:
        pass
    grad_cam = GradCAM(net,args.network, layer_name)
    for i in tqdm(range(args.num)):
        # 输入
        image_path=image_list[i].split(' ')[0]
        idx=image_path.split('/')[-1]
        image_path=os.path.join(args.root,image_path)
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224,224),Image.ANTIALIAS)
        img = np.float32(img) / 255.
        inputs = prepare_input(img)
        inputs = inputs.cuda()

        # 输出图像
        image_dict = {}
        # 网络 
        # ***第一处自定义的地方:自己的模型
        # # #*** 第三处自定义的地方如果网络的输出结果不是单一的，需要在GradCAM修改 
        grad_cam._register_hook() 
        mask = grad_cam(inputs, args.class_id)  # cam mask
        image_dict['cam'], _ = gen_cam(img, mask)
        grad_cam.remove_handlers()


        # #*** 第四处自定义的地方如果网络的输出结果不是单一的，需要在GuidedBackPropagation修改 ，同上
        # # GuidedBackPropagation
        # gbp = GuidedBackPropagation(net,args.network)
        # inputs.grad.zero_()  # 梯度置零
        # grad = gbp(inputs)
        # gb = gen_gb(grad)
        # image_dict['gb'] = norm_image(gb)

        # # # 生成Guided Grad-CAM
        # cam_gb = gb * mask[..., np.newaxis]
        # image_dict['cam_gb'] = norm_image(cam_gb)

        save_image(image_dict, idx, args.network, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_list', type=str, default='/datastore/users/kaibin.tian/DE_experiment/domainnet/clipart_test.txt',
                        help='input image list')
    parser.add_argument('--root', type=str, default='/datastore/users/kaibin.tian/DE_experiment/domainnet/',
                        help='root')
    parser.add_argument('--network', type=str, default='DIS',
                            help='network')
    parser.add_argument('--weight_path', type=str, default='/datastore/users/kaibin.tian/DE_experiment/kd_new/log/KD_DDC/domainnet/[c2r].pth',
                            help='weight_path')
    parser.add_argument('--class-id', type=int, default=None,
                        help='class id')
    parser.add_argument('--output-dir', type=str, default='/datastore/users/kaibin.tian/DE_experiment/grad_cam_result/KD_DDC_c2r_source_cam_results',
                        help='output directory to save results')
    parser.add_argument('--num_class', default=345, type=int,
                    help='the number of classes')
    parser.add_argument('--num', default=1000, type=int,
                    help='pre 1000')
    arguments = parser.parse_args()

    main(arguments)

