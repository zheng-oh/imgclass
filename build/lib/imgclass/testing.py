import torch
import torch.nn as nn
import numpy as np


class Test:
    def __init__(self, device, net_num=18, batch_size=8, num_classes=10,target_layers=['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']):
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.net_num = net_num
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.test_datasets = ""
        self.dataloaders = ""
        self.test_num = 0
        self.labels = []  # 存储标签名称
        self.target_list = []  # 存储真实标签
        self.score_list = []  # 存储预测得分
        self.preds_list = []  # 存储预测标签
        self.matrix = None  # 初始化混淆矩阵
        self.test_loss = 0
        self.test_acc = 0
        self.features_by_layer = {}
        self.target_layers=target_layers

    def _register_hooks(self):
        """注册钩子函数来捕获每层特征"""
        self.hooks = []
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)

    def _hook_fn(self, name):
        """钩子函数捕获特征"""
        def hook(module, input, output):
            # 将每层的输出特征保存到字典中
            if name not in self.features_by_layer:
                self.features_by_layer[name] = []
            self.features_by_layer[name].append(output.detach().cpu().numpy())  # 存储每个 batch 的特征
        return hook

    def confusion_matrix(self, conf_matrix):
        """更新混淆矩阵"""
        if conf_matrix is None:
            conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        
        for p, t in zip(self.preds_list, self.target_list):
            conf_matrix[p, t] += 1  # p 和 t 是整数，表示标签
        return conf_matrix


    def runtest(self):
        print("加载model: {}".format(self.modelname))  
        self._register_hooks()
        self.test_num = len(self.dataloaders['test'].dataset)
        self.runwork()
        result = {"test_acc": self.test_acc, "test_loss": self.test_loss}
        print(
            "test_acc:{:.2f},test_loss{:.2f}".format(
                result["test_acc"], result["test_loss"]
            )
        )

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

