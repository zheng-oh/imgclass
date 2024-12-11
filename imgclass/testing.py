import os.path
import torch
import torch.nn as nn
import numpy as np


class Test:
    def __init__(self, modelname="train", dp="./data", net_num=18, batch_size=8):
        self.criterion = nn.CrossEntropyLoss()
        self.modelname = modelname
        self.dp = dp
        self.net_num = net_num
        self.batch_size = batch_size
        self.num_classes = 0
        self.test_datasets = ""
        self.dataloaders = ""
        self.test_num = 0
        self.lables = []  # 存储标签名称
        self.target_list = []  # 存储真实标签
        self.score_list = []  # 存储预测得分
        self.preds_list = []  # 存储预测标签
        self.matrix = []  # 存储混淆矩阵
        self.test_loss = 0
        self.test_acc = 0

    def runtest(self):
        print("加载model: {}".format(self.modelname))  
        self.test_num = len(self.dataloaders['test'].dataset)
        self.runwork()
        result = {"test_acc": self.test_acc, "test_loss": self.test_loss}
        print(
            "test_acc:{:.2f},test_loss{:.2f}".format(
                result["test_acc"], result["test_loss"]
            )
        )

    def confusion_matrix(self, conf_matrix):
        for p, t in zip(self.preds_list, self.target_list):
            conf_matrix[p, t] += 1
        return conf_matrix

    def runwork(self):
        self.model.eval()
        correct = 0.0
        self.matrix = torch.zeros(self.num_classes, self.num_classes)  # 创建一个空矩阵存储混淆矩阵
        with torch.no_grad():
            for batch_id, (data, target) in enumerate(self.dataloaders['test']):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target).item()
                e_class, preds = torch.max(output, dim=1)
                correct += torch.sum(preds == target)
                self.score_list.extend(output.detach().cpu().numpy())
                self.target_list.extend(target.cpu().numpy())
                self.preds_list.extend(preds.cpu().numpy())
            self.test_loss += loss
            self.test_acc = correct / self.test_num
            self.matrix = self.confusion_matrix(self.matrix)
            self.matrix = self.matrix.cpu()
            self.matrix = np.array(self.matrix)

