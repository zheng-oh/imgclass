import os.path

# import sys
# sys.path.append("..")
import torch
import torch.nn as nn
from prepro import run_pre
from result import Resultplt
import numpy as np


class Test:
    def __init__(self, modelname="train", dp="./data", net_num=50, batch_size=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.modelname = modelname
        self.modelmsg = ""
        self.model = ""
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

    def run(self):
        print("进入testing")
        self.get_model()
        classes = self.get_dataloaders()
        self.lables = classes
        self.num_classes = len(classes)
        print(classes)
        self.pred()
        result = {"test_acc": self.test_acc, "test_loss": self.test_loss}
        print(
            "test_acc:{:.2f},test_loss{:.2f}".format(
                result["test_acc"], result["test_loss"]
            )
        )
        if not os.path.exists("results"):
            os.mkdir("results")
        torch.save(result, "./results/{}.pkl".format(self.modelname))
        self.plt()

    def get_model(self):
        print("加载model：{}".format(self.modelname))
        self.modelmsg = torch.load("./models/{}.pkl".format(self.modelname))
        self.model = self.modelmsg["model"].to(self.device)
        # print(self.modelmsg)

    def get_dataloaders(self):
        t_datasets, self.dataloaders = run_pre(
            self.dp, stage="test", net_num=self.net_num, batch_size=self.batch_size
        )
        self.test_num = len(t_datasets)
        return t_datasets.classes

    def confusion_matrix(self, conf_matrix):
        # print(self.preds_list)
        # print(self.target_list)
        for p, t in zip(self.preds_list, self.target_list):
            conf_matrix[p, t] += 1
        return conf_matrix

    def pred(self):
        self.model.eval()
        correct = 0.0
        self.matrix = torch.zeros(self.num_classes, self.num_classes)  # 创建一个空矩阵存储混淆矩阵
        with torch.no_grad():
            for batch_id, (data, target) in enumerate(self.dataloaders):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target).item()
                e_class, preds = torch.max(output, dim=1)
                print(output)
                print(output.shape)
                print(e_class)
                print(preds)
                correct += torch.sum(preds == target)
                self.score_list.extend(output.detach().cpu().numpy())
                self.target_list.extend(target.cpu().numpy())
                self.preds_list.extend(preds.cpu().numpy())
            self.test_loss += loss
            self.test_acc = correct / self.test_num
            self.matrix = self.confusion_matrix(self.matrix)
            self.matrix = self.matrix.cpu()
            self.matrix = np.array(self.matrix)
            # corrects = self.matrix.diagonal(offset=0)
            # per_kinds = self.matrix.sum(axis=1)
            # print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.size(self.matrix)), self.test_num))
            # 获取每种Emotion的识别准确率
            # print("每种类别总个数：", per_kinds)
            # print("每种类别预测正确的个数：", corrects)
            # print("每种类别的识别准确率为：{0}".format([rate * 100 for rate in corrects / per_kinds]))

    def plt(self):
        p = Resultplt(
            self.num_classes,
            self.target_list,
            self.score_list,
            self.matrix,
            self.lables,
            self.test_acc,
        )
        # p.roc("roc_{}".format(self.modelname))
        # p.conf_mat("matrix_{}".format(self.modelname))
        p.max_roc("max_roc_{}".format(self.modelname))
