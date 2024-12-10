# 数据预处理模块
import torch
from torchvision import datasets
from torchvision.transforms import transforms
from PIL import Image


def run_pre(data_path, stage="train", normalize=False):
    # 定义基础的图像预处理步骤
    base_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]
    
    # 可选的标准化操作，用于将像素值归一化
    if normalize:
        base_transforms.append(
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
    
    # 根据不同阶段添加特定的数据增强操作
    if stage == "train":
        additional_transforms = [
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ]
        data_transforms = transforms.Compose(additional_transforms + base_transforms)
        the_datasets = datasets.ImageFolder(data_path, data_transforms)
    elif stage == "test":
        data_transforms = transforms.Compose(base_transforms)
        the_datasets = datasets.ImageFolder(data_path, data_transforms)
    elif stage == "pred":
        data_transforms = transforms.Compose(base_transforms)
        image = Image.open(data_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        the_datasets = data_transforms(image).unsqueeze(0)
    else:
        raise ValueError("Stage must be 'train', 'test' or 'pred'.")

    return the_datasets
