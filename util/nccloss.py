import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class NCCLoss(nn.Module):
    def __init__(self):
        super(NCCLoss, self).__init__()

    def forward(self,img1,img2):
        """
        计算两张图像之间的归一化互相关（NCC）值。
        img1 和 img2 必须具有相同的形状。
        """
        # 确保输入形状一致
        assert img1.shape == img2.shape, "Input images must have the same shape"

        # 计算每个图像的均值和标准差
        img1_mean = img1.mean(dim=(0, 1), keepdim=True)
        img2_mean = img2.mean(dim=(0, 1), keepdim=True)

        img1_std = img1.std(dim=(0, 1), keepdim=True) + 1e-8
        img2_std = img2.std(dim=(0, 1), keepdim=True) + 1e-8

        # 计算去均值后的图像
        img1_zero_mean = img1 - img1_mean
        img2_zero_mean = img2 - img2_mean

        # 计算归一化互相关
        numerator = torch.sum(img1_zero_mean * img2_zero_mean, dim=(0, 1))
        denominator = img1_std * img2_std * img1.numel()  # numel() 给出像素数量

        ncc = numerator / denominator
        return 1 - ncc.mean() # 返回均值作为总相似度