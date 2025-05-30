import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import imageio
import scipy.io as sio
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch.optim as optim
from tqdm import tqdm
import SimpleITK as sitk
from model import Uformer
#
from torchsummary import summary
#
#
#
# # ONLY MODIFY SETTING HERE
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
#
# model = Uformer(img_size=512, dd_in=3,embed_dim=16,depths=[3,3, 3, 3, 3,3, 3, 3, 3], win_size=8, token_projection='linear',
#                             token_mlp='leff', modulator=True)
# print("\n======== Network Architecture ========")
# input_shape = (3, 512, 512)
# model = model.to(device)
# summary(model,input_size=input_shape)
# with open("network_architecture.txt", "w") as f:
#     f.write(str(model))
#
#
from io import StringIO
import sys
import csv
def save_model_summary_to_csv(model, input_shape, csv_path):
    # 创建一个文本流来捕获summary的输出

    old_stdout = sys.stdout
    result = StringIO()
    sys.stdout = result

    # 运行model summary

    summary(model, input_size=input_shape)

    # 恢复标准输出
    sys.stdout = old_stdout

    # 获取捕获的输出文本
    summary_str = result.getvalue()
    result.close()

    # 解析输出文本

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['Layer (type)', 'Output Shape', 'Param #'])

        # 解析每一行
        lines = summary_str.split('\n')
        for line in lines:
            if '=' in line or len(line.strip()) == 0:  # 跳过分隔符行和空行
                continue
            # 分割并清理数据
            parts = [x.strip() for x in line.split('  ') if x.strip()]
            if len(parts) >= 3:  # 确保有足够的列
                layer_info = parts[0]
                output_shape = parts[1]
                params = parts[2]
                writer.writerow([layer_info, output_shape, params])


# 使用示例
csv_path = 'model_summary.csv'
input_shape = (3, 512, 512)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = Uformer(img_size=512, dd_in=3, embed_dim=16, depths=[3, 3, 3, 3, 3, 3, 3, 3, 3], win_size=8,
                    token_projection='linear',
                    token_mlp='leff', modulator=True)
model = model.to(device)
save_model_summary_to_csv(model, input_shape, csv_path)