import torch
from torch import nn
from torchvision import models
import logging
from imgclass.prepro import run_pre
import time
from torch.utils.tensorboard import SummaryWriter


class Train:
    def __init__(
        self,
        name="train",
        dp="./data",
        net_num=50,
        num_epochs=200,
        batch_size=8,
        ratio=0.6,
    ):
        self.name = name
        self.dp = dp
        self.net_num = net_num
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.ratio = ratio
        self.logname = "./logs/%s.log" % name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.dataloaders = {}
        self.class_name = []
        self.train_num = 0
        self.vaild_num = 0
        self.model = {}
        self.optimizer = ""
        self.scheduler = ""
        self.criterion = nn.CrossEntropyLoss()
        self.best_model = {
            "model": "",
            "train_acc": 0,
            "train_loss": 0,
            "valid_acc": 0,
            "valid_loss": 0,
        }
        self.writer = SummaryWriter(log_dir=f"./logs/{self.name}")


    def runtrain(self):
        self.init_model()
        self.runwork()

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

    def runwork(self):
        for epoch in range(self.num_epochs):
            logging.info("训练迭代：%d/%d" % (epoch, self.num_epochs))
            train_acc, train_loss = self.train()
            valid_acc, valid_loss = self.vaild()
            # 使用 SummaryWriter 记录训练和验证的损失和准确率
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/valid', valid_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/valid', valid_acc, epoch)
            if (
                train_acc + valid_acc
                > self.best_model["train_acc"] + self.best_model["valid_acc"]
            ):
                self.best_model["train_acc"] = train_acc
                self.best_model["valid_acc"] = valid_acc
                self.best_model["train_loss"] = train_loss
                self.best_model["valid_loss"] = valid_loss
                self.best_model["model"] = self.model
                logging.info(
                    "{} 最优模型:train_acc:{},train_loss:{},valid_acc:{},valid_loss:{}".format(
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        self.best_model["train_acc"],
                        self.best_model["train_loss"],
                        self.best_model["valid_acc"],
                        self.best_model["valid_loss"],
                    )
                )
                torch.save(self.best_model, "./models/{0}.pkl".format(self.name))
            # 训练结束后确保所有日志数据写入磁盘
            self.writer.flush()
        self.writer.close()
        logging.info(
            "{} Finished".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        )

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
                # 记录到 TensorBoard
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
