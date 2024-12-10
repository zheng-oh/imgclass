import os.path
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms


class Pred:
    def __init__(self, modelname="train"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.modelname = modelname
        self.model = None

    def runpred(self):
        print("加载model: {}".format(self.modelname))
        self.modelmsg = torch.load("./models/{}.pkl".format(self.modelname))
        self.model = self.modelmsg["model"].to(self.device)    
        pred_class, confidence = self.runwork()
        return pred_class, confidence
    def runwork(self):
        self.model.eval()
        # 进行预测
        with torch.no_grad():
            output = self.model(self.dataloaders['pred'])
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, pred_class = torch.max(probabilities, dim=1)
            return pred_class, confidence