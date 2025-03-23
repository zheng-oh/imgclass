#trainging.py
import torch
from torch import nn
import logging
from imgclass.prepro import run_pre
import time
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

class Train:
    def __init__(
        self,
        name="train",
        dp="./data",
        net_num=50,
        num_epochs=200,
        batch_size=8,
        ratio=0.6,  # 用于非 K 折时的训练-验证划分比例
        k_folds=5,  # K 折交叉验证的折数，设为 None 时不使用 K 折
        device="cpu",
    ):
        self.name = name
        self.dp = dp
        self.net_num = net_num
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.ratio = ratio
        self.k_folds = k_folds  # 如果为 None，则不使用 K 折
        self.device = device
        self.dataloaders = {}
        self.class_name = []
        self.train_num = 0
        self.valid_num = 0
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        self.best_model = {
            "model": None,
            "train_acc": 0,
            "train_loss": 0,
            "valid_acc": 0,
            "valid_loss": 0,
        }
        self.kfold_results = []  # 仅在 K 折时使用


    def runtrain(self):
        if self.k_folds is None or self.k_folds <= 1:  # 不使用 K 折
            self.init_model()
            self.runwork(fold=None)  # 单次训练，不传入 fold 参数
        else:  # 使用 K 折
            for fold, dataloaders in enumerate(self.fold_dataloaders):
                logging.info(f"Starting fold {fold + 1}/{self.k_folds}")
                self.dataloaders = dataloaders
                self.train_num = dataloaders["train_num"]
                self.valid_num = dataloaders["valid_num"]

                # 每次折叠重新初始化模型
                self.init_model()
                self.runwork(fold)

                # 记录当前折叠的最优结果
                self.kfold_results.append({
                    "fold": fold + 1,
                    "train_acc": self.best_model["train_acc"],
                    "valid_acc": self.best_model["valid_acc"],
                    "train_loss": self.best_model["train_loss"],
                    "valid_loss": self.best_model["valid_loss"],
                    "model": self.best_model["model"],  # 添加模型以备后用
                })
            self.summarize_kfold_results()

    def runwork(self, fold):
        for epoch in range(self.num_epochs):
            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            if fold is None:
                logging.info(f"{current_time} - Epoch {epoch + 1}/{self.num_epochs}")
            else:
                logging.info(f"{current_time} - Fold {fold + 1} - Epoch {epoch + 1}/{self.num_epochs}")
            train_acc, train_loss = self.train()
            valid_acc, valid_loss = self.valid()

            # 使用 SummaryWriter 记录训练和验证的损失和准确率
            tag_prefix = "" if fold is None else f"fold_{fold + 1}_"
            self.writer.add_scalar(f'Loss/train_{tag_prefix}', train_loss, epoch)
            self.writer.add_scalar(f'Loss/valid_{tag_prefix}', valid_loss, epoch)
            self.writer.add_scalar(f'Accuracy/train_{tag_prefix}', train_acc, epoch)
            self.writer.add_scalar(f'Accuracy/valid_{tag_prefix}', valid_acc, epoch)

            # 检查是否为最优模型并实时保存
            if (train_acc + valid_acc > self.best_model["train_acc"] + self.best_model["valid_acc"]):
                self.best_model["train_acc"] = train_acc.item()
                self.best_model["valid_acc"] = valid_acc.item()
                self.best_model["train_loss"] = train_loss.item()
                self.best_model["valid_loss"] = valid_loss.item()
                self.best_model["model"] = self.model

                # 实最优模型时保存
                save_path = f"./models/{self.name}.pth"
                torch.save(self.model.state_dict(), save_path)
                log_msg = (
                    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
                    f"{'Single' if fold is None else f'Fold {fold + 1}'} 最优模型已保存到 {save_path}: "
                    f"train_acc:{self.best_model['train_acc']:.4f}, "
                    f"train_loss:{self.best_model['train_loss']:.4f}, "
                    f"valid_acc:{self.best_model['valid_acc']:.4f}, "
                    f"valid_loss:{self.best_model['valid_loss']:.4f}"
                )
                logging.info(log_msg)
            self.scheduler.step()
        self.writer.flush()

    def train(self):
        self.model.train()
        self.model = self.model.to(self.device)
        correct = 0.0
        total_loss = 0.0
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
        return accuracy, total_loss

    def valid(self):
        self.model.eval()
        correct = 0.0
        total_loss = 0.0
        with torch.no_grad():
            for batch_id, (data, target) in enumerate(self.dataloaders["valid"]):
                data, target = data.to(self.device), target.to(self.device)
                self.model = self.model.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss
                _, preds = torch.max(output, dim=1)
                correct += torch.sum(preds == target)
        accuracy = correct / self.valid_num
        return accuracy, total_loss

    def summarize_kfold_results(self):
        avg_train_acc = sum(r["train_acc"] for r in self.kfold_results) / self.k_folds
        avg_valid_acc = sum(r["valid_acc"] for r in self.kfold_results) / self.k_folds
        avg_train_loss = sum(r["train_loss"] for r in self.kfold_results) / self.k_folds
        avg_valid_loss = sum(r["valid_loss"] for r in self.kfold_results) / self.k_folds

        logging.info(f"K-Fold Cross-Validation Summary ({self.k_folds} folds):")
        logging.info(f"Average Train Accuracy: {avg_train_acc:.4f}")
        logging.info(f"Average Valid Accuracy: {avg_valid_acc:.4f}")
        logging.info(f"Average Train Loss: {avg_train_loss:.4f}")
        logging.info(f"Average Valid Loss: {avg_valid_loss:.4f}")

        best_fold = max(self.kfold_results, key=lambda x: x["valid_acc"])
        logging.info(f"Best Fold: {best_fold['fold']} with Valid Acc: {best_fold['valid_acc']:.4f}")

# # 测试代码
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 使用 K 折
#     trainer = Train(name="my_kfold_train", dp="./data/train", net_num=50, num_epochs=10, batch_size=8, k_folds=5, device=device)
#     trainer.get_dataloaders()
#     trainer.runtrain()
    
#     # 不使用 K 折
#     trainer = Train(name="my_single_train", dp="./data/train", net_num=50, num_epochs=10, batch_size=8, k_folds=None, device=device)
#     trainer.get_dataloaders()
#     trainer.runtrain()