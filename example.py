import os
import torch
import json
import logging
import time
from torchvision import models
from torch import nn
from imgclass.training import Train
from imgclass.testing import Test
from imgclass.prepro import run_pre
from imgclass.pred import Pred
from imgclass.result import Resultplt

from torch.utils.tensorboard import SummaryWriter

# 创建 Deepnet 类的实例
class Deeptrain(Train):
    def __init__(self,
        name="mywork",    # 自定义任务名称
        dp="./data",       # 数据目录
        net_num=18,        # 网络模型层数可选18或者50代表ResNet50，ResNet50
        num_epochs=200,    # 训练迭代轮次
        batch_size=8,     # 批处理大小
        ratio=0.8,         #训练和 验证集的比例3:2
        device="cpu",
    ):
        self.device = device
        # 检查 net_num 是否为 18 或 50
        if net_num not in [18, 50]:
            raise ValueError("net_num must be 18 or 50")
        super().__init__(name, dp, net_num, num_epochs, batch_size, ratio)
    #
    
    def run(self):
        self.get_dataloaders()
        self.init_model()
        self.runtrain()
        self.save_result()
        logging.info(
            "{} Finished".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        )

    def get_dataloaders(self):
        datasets = {}
        image_datasets = run_pre(self.dp, stage="train", normalize=False) #关键 获得预处理数据
        self.class_name = image_datasets.classes
        self.name = self.name + "_" + str(len(self.class_name))
        self.writer = SummaryWriter(log_dir=f"./runs/{self.name}")
        samp_num = len(image_datasets)
        self.train_num, self.vaild_num = int(self.ratio * samp_num), samp_num - int(
            self.ratio * samp_num
        )
        datasets["train"], datasets["valid"] = torch.utils.data.random_split(
            image_datasets, [self.train_num, self.vaild_num]
        )
        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                dataset=datasets[x], batch_size=self.batch_size, shuffle=True
            )
            for x in ["train", "valid"]
        }

    def init_model(self):
        if self.net_num == 50:
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif self.net_num == 18:
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            print("输入的模型层数错误")
            return
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, len(self.class_name)),
            nn.LogSoftmax(dim=1),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1
        )

    def save_result(self):
        torch.save(self.best_model["model"].state_dict(), "./models/{0}.pth".format(self.name))
        # 读取 JSON 文件并更新数据
        try:
            with open('./results/models_info.json', 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}  # 如果文件不存在或者为空，则初始化为空字典
        # 更新信息
        if self.name not in data:
            data[self.name] = {}
        data[self.name]['train_acc'] = self.best_model["train_acc"]
        data[self.name]['valid_acc'] = self.best_model["valid_acc"]
        data[self.name]['train_loss'] = self.best_model["train_loss"]
        data[self.name]['valid_loss'] = self.best_model["valid_loss"]
        data[self.name]['epochs'] = self.num_epochs
        data[self.name]['ratio'] = self.ratio
        data[self.name]['class_name'] = self.class_name
        # 将更新后的数据写回 JSON 文件
        with open('./results/models_info.json', 'w') as f:
            json.dump(data, f, indent=4)  # 使用 indent 使 JSON 格式化更美观


class Deeptest(Test):
    def __init__(self,
        modelname="mymodel",    # 选择模型
        dp="./data",       # 测试数据目录
        net_num=18,        # 网络模型层数可选18或者50代表ResNet50，ResNet50
        batch_size=8,     # 批处理大小
        device="cpu",
    ):
        self.modelname = modelname
        self.dp = dp
        self.device = device
        # 检查 net_num 是否为 18 或 50
        if net_num not in [18, 50]:
            raise ValueError("net_num must be 18 or 50")
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) if net_num == 50 else models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        super().__init__(device,net_num, batch_size)
    #
    def run(self):
        self.get_dataloaders()
        self.init_model()
        print(self.model)
        self.runtest()
        self.plt()
        print(
            "{} Finished".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        )

    def get_dataloaders(self):
        image_datasets = run_pre(self.dp, stage="test", normalize=False) #关键 获得预处理数据
        self.class_name = image_datasets.classes
        self.modelname = self.modelname + "_" + str(len(self.class_name))
        self.dataloaders = {"test": torch.utils.data.DataLoader(dataset=image_datasets, batch_size=self.batch_size, shuffle=True)
        }
        print(f"Number of test samples: {len(image_datasets)}")
        self.lables = image_datasets.classes
        self.num_classes = len(self.lables)

    def init_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, len(self.class_name)),
            nn.LogSoftmax(dim=1),
        )
        # 加载模型权重
        self.model.load_state_dict(torch.load(f"./models/{self.modelname}.pth",weights_only=True))
        self.model.to(self.device)

    def plt(self):
        p = Resultplt(
            self.num_classes,
            self.target_list,
            self.score_list,
            self.matrix,
            self.lables,
            self.test_acc,
            self.preds_list,
            self.features_by_layer,
        )
        p.max_roc("max_roc_{}".format(self.modelname))
        p.tsne("tsne_{}".format(self.modelname))

class Deeppred(Pred):
    def __init__(self,  
        device="cpu",
    ):
        self.dataloaders = {}
        self.device = device
        super().__init__()

    def load_model(self, modelname,class_num):
        if "resnet50" in modelname.lower():
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif "resnet18" in modelname.lower():
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            raise ValueError("Model not found")
        for param in self.model.parameters():
            param.requires_grad = False
       # 修改全连接层 (fc) 输出为类别数，并将模型移动到指定设备
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, class_num),
            nn.LogSoftmax(dim=1)
        )
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


def main(work,name):
    net_num=18 #网络层数
    ratio = 0.8 #训练和验证集的比例
    epochs = 200 #训练迭代轮次
    for initdir in ['logs','models','results']:
        if not os.path.exists(initdir):
            os.makedirs(initdir)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logging.basicConfig(filename="./logs/{0}_Resnet{1}.log".format(name,net_num), level=logging.INFO)
    logging.info(
    "{} Started {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),work)
    )
    logging.info(
        "Device: {}".format(device)
    )
    if work=="train":
        dp = "./data/{}/train".format(name)
        formatname = "{0}_Resnet{1}".format(name, net_num)
        tr = Deeptrain(name=formatname, dp=dp, net_num=net_num, ratio=ratio,num_epochs=epochs,device=device)
        tr.run()
    elif work == "test":
        dp = "./data/{}/test".format(name)
        modelname = "{0}_Resnet{1}".format(name, net_num)
        te = Deeptest(modelname=modelname,dp=dp,net_num=net_num, device=device)
        te.run()
    elif work == "pred":
        modelname = name  #注意！！！！ 模型名称,在 models 目录下
        dp = "./test.png" #测试数据路径
        pred = Deeppred(device=device)
        pred.load_model(modelname, int(modelname.split("_")[-1]))
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
    main(works[1], 'food') 
    # main(works[1], 'food')
    # main(works[2], 'food_Resnet18_4') 
    
