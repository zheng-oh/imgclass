import torch
import torch.nn as nn
from prepro import run_pre


class Pred:
    def __init__(self, modelname: "train", dp, net_num=50, batch_size=8):
        self.modelname = modelname
        self.dp = dp
        self.net_num = net_num
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.model = ''
        self.best_model = {}
        self.test_datasets = ''
        self.dataloaders = ''

    def run(self):
        self.best_model = torch.load('./models/{}.pkl'.format(self.modelname))
        self.model = self.best_model["model"].to(self.device)
        self.test_datasets, self.dataloaders = run_pre(self.dp, stage="train", net_num=self.net_num)
        self.pred()

    def pred(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0.0
        totel_test = 0.0
        with torch.no_grad():
            for batch_id, (data, target) in enumerate(self.dataloaders):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # print(output)
                total_loss += self.criterion(output, target).item()
                _, preds = torch.max(output, dim=1)
                print(output)
                print(preds)
        #         correct += torch.sum(preds == target)
        #         totel_test +=len(data)
        #     accuracy = correct / totel_test
        #     print("Test_Loss:{:.4f}".format(accuracy))
        #     print("Test_Accuracy:{:.4f}".format(total_loss))
