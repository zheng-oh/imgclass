import torch
from torch import nn
from torchvision import models
import logging
from prepro import run_pre
import time


class Train:
    def __init__(self, name= "train", dp = "./data", net_num= 50, num_epochs=200, batch_size=8, ratio=0.6):
        self.name = name
        self.dp = dp
        self.net_num = net_num
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.ratio = ratio
        self.logname = './logs/%s.log' % name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataloaders = {}
        self.class_num = 0
        self.train_num = 0
        self.vaild_num = 0
        self.model = {}
        self.optimizer = ''
        self.scheduler = ''
        self.criterion = nn.CrossEntropyLoss()
        self.best_model = {"model": '', "train_acc": 0, "train_loss": 0, "valid_acc": 0, "valid_loss": 0}

    def run(self):
        logging.basicConfig(filename=self.logname, level=logging.INFO)
        logging.info('{} Started'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        self.dataloaders, self.class_num, self.train_num, self.vaild_num = self.get_dataloaders()
        self.model = self.get_model()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        for epoch in range(self.num_epochs):
            logging.info("训练迭代：%d/%d" % (epoch, self.num_epochs))
            train_acc, train_loss = self.train()
            valid_acc, valid_loss = self.vaild()
            if train_acc + valid_acc > self.best_model["train_acc"] + self.best_model["valid_acc"]:
                self.best_model["train_acc"] = train_acc
                self.best_model["valid_acc"] = valid_acc
                self.best_model["train_loss"] = train_loss
                self.best_model["valid_loss"] = valid_loss
                self.best_model["model"] = self.model
                logging.info("{} 最优模型:train_acc:{},train_loss:{},valid_acc:{},valid_loss:{}".format(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), self.best_model["train_acc"],
                    self.best_model["train_loss"], self.best_model["valid_acc"], self.best_model["valid_loss"]))
                torch.save(self.best_model, "./models/{0}.pkl".format(self.name))
        logging.info('{} Finished'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    def get_dataloaders(self):
        datasets = {}
        image_datasets, image_dataloaders = run_pre(self.dp, stage="train", net_num=self.net_num)
        samp_num = len(image_datasets)
        train_num, vaild_num = int(self.ratio * samp_num), samp_num - int(self.ratio * samp_num)
        datasets["train"], datasets["valid"] = torch.utils.data.random_split(image_datasets, [train_num, vaild_num])
        dataloaders = {
            x: torch.utils.data.DataLoader(
                dataset=datasets[x], batch_size=self.batch_size, shuffle=True
            )
            for x in ["train", "valid"]
        }
        return dataloaders, image_datasets.classes, train_num, vaild_num

    def get_model(self):
        if self.net_num == 50:
            model_pre = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif self.net_num == 18:
            model_pre = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            print("输入的模型层数错误")
            return
        for param in model_pre.parameters():
            param.requires_grad = False
        print(model_pre.fc.in_features)
        model_pre.fc = nn.Sequential(
            nn.Linear(model_pre.fc.in_features, len(self.class_num)),
            nn.LogSoftmax(dim=1),
        )
        return model_pre

    def train(self):
        self.model.train()
        self.model = self.model.to(self.device)
        correct = 0.0
        total_loss = 0.0  # 初始化
        for batch_id, (data, target) in enumerate(self.dataloaders["train"]):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss
            _, preds = torch.max(output, dim=1)
            correct += torch.sum(preds == target)
        accuracy = correct / self.train_num
        self.scheduler.step()
        return accuracy, total_loss

    def vaild(self):
        self.model.eval()
        correct = 0.0
        total_loss = 0.0
        with torch.no_grad():
            for batch_id, (data, target) in enumerate(self.dataloaders["valid"]):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target).item()
                _, preds = torch.max(output, dim=1)
                correct += torch.sum(preds == target)
            total_loss += loss
            accuracy = correct / self.vaild_num
            return accuracy, total_loss
