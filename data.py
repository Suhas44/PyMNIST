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

    training_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    return training_dataloader

def get_test_dataloader():
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=ToTensor()
    )
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    return test_dataloader