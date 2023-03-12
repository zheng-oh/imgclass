import torch
from torchvision import datasets
from torchvision.transforms import transforms


def run_pre(data_path, stage="train", net_num=50, batch_size=8):
    if stage == "train":
        data_transforms = transforms.Compose(
            [
                transforms.Resize(256,),
                transforms.CenterCrop(224),
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ) if net_num == "50" else transforms.Compose(
            [
                transforms.Resize(256,),
                transforms.CenterCrop(224),
                transforms.Resize(64,),
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    elif stage == "test":
        data_transforms = transforms.Compose(
            [
                transforms.Resize(256,),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ) if net_num == "50" else transforms.Compose(
            [
                transforms.Resize(256,),
                transforms.CenterCrop(224),
                transforms.Resize(64,),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    the_datasets = datasets.ImageFolder(data_path, data_transforms)
    dataloaders = torch.utils.data.DataLoader(dataset=the_datasets, batch_size=batch_size, shuffle=True)
    return the_datasets, dataloaders
