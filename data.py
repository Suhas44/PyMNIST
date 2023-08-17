import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def download_dataset():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

def get_training_dataloader():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=False,
        transform=ToTensor()
    )
    train_subset, val_subset = torch.utils.data.random_split(
        training_data, [50000, 10000], generator=torch.Generator().manual_seed(1))
    training_dataloader = DataLoader(dataset=train_subset, shuffle=True, batch_size=64)
    validation_dataloader = DataLoader(dataset=val_subset, shuffle=False, batch_size=64)
    return training_dataloader, validation_dataloader

def get_test_dataloader():
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=ToTensor()
    )
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    return test_dataloader