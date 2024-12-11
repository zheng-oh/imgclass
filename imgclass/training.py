import torch
from torch import nn
import logging
from imgclass.prepro import run_pre
import time


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


    def runtrain(self):
        self.runwork()


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
                self.best_model["train_acc"] = train_acc.item()
                self.best_model["valid_acc"] = valid_acc.item()
                self.best_model["train_loss"] = train_loss.item()
                self.best_model["valid_loss"] = valid_loss.item()
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
            # 训练结束后确保所有日志数据写入磁盘
            self.writer.flush()
        self.writer.close()


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
                loss = self.criterion(output, target)
                _, preds = torch.max(output, dim=1)
                correct += torch.sum(preds == target)
            total_loss += loss
            accuracy = correct / self.vaild_num
            return accuracy, total_loss
