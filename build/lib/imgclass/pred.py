import io
import base64
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')  # 设置后端为非交互式
import matplotlib.pyplot as plt

class FeatureMapVisualizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.features = []

    def register_hooks(self):
        """注册钩子，用于捕获中间层的输出"""
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):  # 只对卷积层进行操作
                layer.register_forward_hook(self.save_features)

    def save_features(self, module, input, output):
        """保存特征图"""
        self.features.append(output)

    def plot_feature_maps(self):
        """绘制每层的第一个特征图，且合并为一个图像"""
        # 计算合适的网格大小
        num_features = len(self.features)
        grid_size = int(num_features**0.5) + 1  # 创建合适大小的网格
        plt.figure(figsize=(grid_size * 2, grid_size * 2))
        for idx, feature_map in enumerate(self.features):
            # 选择该层的第一个特征图
            feature = feature_map[0, -1].cpu().detach().numpy()  # 取第一个batch的最后一个通道
            plt.subplot(grid_size, grid_size, idx + 1)  # 根据特征图数量确定网格位置
            plt.imshow(feature, cmap='gray')
            plt.axis('off')  # 关闭坐标轴
        # 将合并后的特征图保存为BytesIO对象
        img_buf = io.BytesIO()
        plt.tight_layout()  # 调整子图间的距离
        plt.savefig(img_buf, format='png')
        plt.close()  # 关闭当前图形
     # 将图片转换为Base64编码
        img_buf.seek(0)
        return base64.b64encode(img_buf.read()).decode('utf-8')
class Pred:
    def __init__(self, modelname="train"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.modelname = modelname
        self.model = None
        self.feature_map_visualizer = None

    def runpred(self):
        self.modelmsg = torch.load("./models/{}.pkl".format(self.modelname))
        self.model = self.modelmsg["model"].to(self.device)    
                # 注册钩子来捕获特征图
        self.feature_map_visualizer = FeatureMapVisualizer(self.model, self.device)
        self.feature_map_visualizer.register_hooks()
        
        # 获取预测结果
        pred_class, confidence = self.runwork()
        # 绘制特征图
        self.feature_map = self.feature_map_visualizer.plot_feature_maps()
        return pred_class, confidence

    def runwork(self):
        self.model.eval()
        # 进行预测
        with torch.no_grad():
            output = self.model(self.dataloaders['pred'])
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, pred_class = torch.max(probabilities, dim=1)
            return pred_class, confidence