import os
import torch
import json
import logging
import time
from torchvision import models
from torch import nn
import numpy as np

from imgclass.training import Train
from imgclass.testing import Test
from imgclass.prepro import run_pre,get_dataloaders
from imgclass.pred import Pred
from imgclass.result import Resultplt
from torch.utils.tensorboard import SummaryWriter

def build_model(net_num, num_classes, mode='train'):
    if net_num == 50:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif net_num == 18:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        raise ValueError("net_num must be 18 or 50")

    if mode == 'train':
        # 训练模式：解冻layer
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif mode in ['test', 'pred']:
        # 测试/预测模式：所有参数都冻结
        for param in model.parameters():
            param.requires_grad = False

    # 重新定义全连接层
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
        nn.LogSoftmax(dim=1),
    )
    return model


# 创建 Deeptrain 类的实例，支持 K 折交叉验证
class Deeptrain(Train):
    def __init__(self,
        name="mywork",    # 自定义任务名称
        dp="./data",       # 数据目录
        net_num=18,        # 网络模型层数可选18或者50代表ResNet50，ResNet50
        num_epochs=200,    # 训练迭代轮次
        batch_size=8,     # 批处理大小
        ratio=0.8,         # 训练和验证集的比例（仅用于非 K 折时）
        k_folds=None,      # 新增：K 折交叉验证的折数，默认 None 表示不使用 K 折
        device="cpu",
    ):
        self.device = device
        if net_num not in [18, 50]:
            raise ValueError("net_num must be 18 or 50")
        super().__init__(name, dp, net_num, num_epochs, batch_size, ratio)
        self.k_folds = k_folds  # K 折参数
        self.fold_dataloaders = []  # 存储每一折的 dataloaders
        self.kfold_results = []  # 存储每一折的结果

    def run(self):
        self.dataloaders = get_dataloaders(self.dp, "train", self.batch_size, self.ratio, self.k_folds)
        if self.k_folds:
            self.fold_dataloaders = self.dataloaders
            self.class_name = self.fold_dataloaders[0]["class_name"]
        else:
            self.class_name = self.dataloaders["class_name"]
        self.name = f"{self.name}_{len(self.class_name)}"
        self.writer = SummaryWriter(log_dir=f"./runs/{self.name}")
        self.runtrain()
        self.save_result()
        logging.info(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} Finished")
   

    def init_model(self):
        self.model = build_model(self.net_num, len(self.class_name), mode='train')
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5
        )

    def save_result(self):
        os.makedirs("./results", exist_ok=True)
        # 只保存训练结果到 JSON，不再保存模型（已在 runwork 中实时保存）
        try:
            with open('./results/models_info.json', 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        data.setdefault(self.name, {}).update({
            'train_acc': self.best_model["train_acc"],
            'valid_acc': self.best_model["valid_acc"],
            'train_loss': self.best_model["train_loss"],
            'valid_loss': self.best_model["valid_loss"],
            'epochs': self.num_epochs,
            'ratio': self.ratio,
            'class_name': self.class_name,
            'k_folds': self.k_folds
        })
        with open('./results/models_info.json', 'w') as f:
            json.dump(data, f, indent=4)

class Deeptest(Test):
    def __init__(self,
        modelname="mymodel",    # 选择模型
        dp="./data",       # 测试数据目录
        net_num=18,        # 网络模型层数可选18或者50代表ResNet50，ResNet50
        batch_size=8,     # 批处理大小
        device="cpu",
    ):
        super().__init__(device,net_num, batch_size)
        self.modelname = modelname
        self.dp = dp
        self.device = device
    #
    def run(self):
        data = get_dataloaders(self.dp, "test", self.batch_size)
        self.dataloaders = {"test": data["loader"]}
        self.class_name = data["class_name"]
        self.num_classes = len(self.class_name)  # 确保 num_classes 正确
        self.labels = self.class_name
        self.modelname = f"{self.modelname}_{self.num_classes}"
        self.init_model()
        self.runtest()
        self.plt()
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} Finished")


    def init_model(self):
        # 不再冻结所有参数，因为训练时解冻了 layer4
        self.model = build_model(self.net_num, len(self.class_name), mode='test')
        # 加载模型权重
        self.model.load_state_dict(torch.load(f"./models/{self.modelname}.pth", weights_only=True))
        self.model.to(self.device)

    def plt(self):
        p = Resultplt(
            self.num_classes,
            self.target_list,
            self.score_list,
            self.matrix,
            self.labels,
            self.test_acc,
            self.preds_list,
            self.features_by_layer,
        )
        p.max_roc("max_roc_{}".format(self.modelname))
        p.tsne("tsne_{}".format(self.modelname))

class Deeppred(Pred):
    def __init__(self,
        net_num=18,        # 网络模型层数可选18或者50代表ResNet50，ResNet50
        device="cpu",
    ):
        self.net_num = net_num
        self.dataloaders = {}
        self.device = device
        super().__init__()

    def load_model(self, modelname,class_num):
        self.class_name = [str(i) for i in range(class_num)]  # 假设类名是数字
        self.model = build_model(self.net_num, len(self.class_name), mode='pred')
        # 加载模型权重
        self.model.load_state_dict(torch.load(f"./models/{modelname}.pth",weights_only=True))
        self.model.to(self.device)

    def predict(self, image):
        # 处理输入图像
        self.dataloaders['pred'] = run_pre(image, stage="pred", normalize=False).to(self.device)
        # 进行预测
        pred_class, confidence = self.runpred()

        # 清理内存
        del self.dataloaders['pred']
        return pred_class, confidence


def main(work,name,num_classes=4):
    net_num=50 #网络层数
    ratio = 0.8 #训练和验证集的比例
    epochs = 100 #训练迭代轮次
    k_folds = 5   # 设置 K 折交叉验证（可根据需要调整或设为 None）
    for initdir in ['logs','models','results']:
        os.makedirs(initdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    log_filename = f"./logs/{name}_Resnet{net_num}{'_kfolds' if k_folds else ''}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO)
    logging.info(
    "{} Started {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),work)
    )
    logging.info(
        "Device: {}".format(device)
    )
    basename = f"{name}_Resnet{net_num}{'_best_fold' if k_folds else ''}"
    if work=="train":
        dp = "./data/{}/train".format(name)
        tr = Deeptrain(name=basename, dp=dp, net_num=net_num, ratio=ratio, num_epochs=epochs, k_folds=k_folds, device=device)        
        tr.run()
    elif work == "test":
        dp = f"./data/{name}/test"
        te = Deeptest(modelname=basename,dp=dp,net_num=net_num, device=device)
        te.run()
    elif work == "pred":
        modelname = f'{basename}_{num_classes}'  #注意！！！！ 模型名称,在 models 目录下
        dp = "./test.jpg" #测试数据路径
        pred = Deeppred(net_num,device=device)
        pred.load_model(modelname, num_classes)
        pred_class, confidence = pred.predict(dp)
        pred_class = pred_class.item()  # 获取数字
        confidence = round(confidence.item(), 2)  # 获取数字并保留2位小数
        print(f"预测类别: {pred_class}, 置信度: {confidence}")
        pred.save_feature_map(modelname)

    else:
        raise ValueError("请输入1、2、3 代表执行训练、测试、预测，'{}' 无效！".format(work))

if __name__ == "__main__":
    works=["train","test","pred"]
    #请输入0、1、2 代表执行训练、测试、外部预测。进行外部预测时，必须使用模型名称
    main(works[0], 'food') 
    # main(works[1], 'food')
    # main(works[2], 'food',4) 
    
