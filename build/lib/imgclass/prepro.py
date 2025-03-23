# 数据预处理模块
import torch
from torchvision import datasets
from torchvision.transforms import transforms
from PIL import Image
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import logging

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
        # 判断data_path是否为图片路径 还是PIL图像
        if isinstance(data_path, str):
            data_path = Image.open(data_path)
        if data_path.mode != 'RGB':
            data_path = data_path.convert('RGB')
        the_datasets = data_transforms(data_path).unsqueeze(0)
    else:
        raise ValueError("Stage must be 'train', 'test' or 'pred'.")

    return the_datasets

def get_dataloaders(dp, stage, batch_size, ratio=0.8, k_folds=None):
    """
    通用的数据加载器生成函数，支持训练、测试和预测阶段。
    
    参数:
        dp (str): 数据路径
        stage (str): 'train', 'test', 或 'pred'，指定数据加载模式
        batch_size (int): 批处理大小
        ratio (float, optional): 训练-验证集分割比例，仅在 stage='train' 且 k_folds=None 时使用
        k_folds (int, optional): K 折交叉验证的折数，仅在 stage='train' 时使用，None 表示不使用 K 折
    
    返回:
        dict: 根据 stage 返回相应的数据加载器和相关信息
    """
    # 根据 stage 加载数据
    if stage not in ['train', 'test', 'pred']:
        raise ValueError("stage must be 'train', 'test', or 'pred'")

    image_datasets = run_pre(dp, stage=stage, normalize=False)
    
    if stage == 'train':
        class_name = image_datasets.classes if hasattr(image_datasets, 'classes') else None
        dataset_size = len(image_datasets)
        
        if k_folds is None or k_folds <= 1:
            # 单次训练-验证分割
            logging.info("Using single train-valid split (no K-fold)")
            train_num = int(ratio * dataset_size)
            valid_num = dataset_size - train_num
            datasets = {}
            datasets["train"], datasets["valid"] = torch.utils.data.random_split(
                image_datasets, [train_num, valid_num]
            )
            dataloaders = {
                "train": DataLoader(
                    datasets["train"], batch_size=batch_size, shuffle=True
                ),
                "valid": DataLoader(
                    datasets["valid"], batch_size=batch_size, shuffle=True
                ),
                "class_name": class_name,
                "train_num": train_num,
                "valid_num": valid_num
            }
            return dataloaders
        else:
            # K 折交叉验证
            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_dataloaders = []
            for fold, (train_idx, valid_idx) in enumerate(kfold.split(range(dataset_size))):
                logging.info(f"Preparing fold {fold + 1}/{k_folds}")
                train_sampler = SubsetRandomSampler(train_idx)
                valid_sampler = SubsetRandomSampler(valid_idx)

                train_loader = DataLoader(
                    image_datasets,
                    batch_size=batch_size,
                    sampler=train_sampler,
                    shuffle=False,
                )
                valid_loader = DataLoader(
                    image_datasets,
                    batch_size=batch_size,
                    sampler=valid_sampler,
                    shuffle=False,
                )

                fold_dataloaders.append({
                    "train": train_loader,
                    "valid": valid_loader,
                    "train_num": len(train_idx),
                    "valid_num": len(valid_idx),
                    "class_name": class_name
                })
            return fold_dataloaders

    elif stage == 'test':
        # 测试模式
        class_name = image_datasets.classes if hasattr(image_datasets, 'classes') else None
        loader = DataLoader(image_datasets, batch_size=batch_size, shuffle=True)
        return {
            "loader": loader,
            "class_name": class_name
        }
    elif stage == 'pred':
        # 预测模式，直接返回预处理后的数据（单张图像）
        return image_datasets  # run_pre 已返回预处理后的单张图像数据