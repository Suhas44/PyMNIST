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

def get_dataloaders(training_data, test_data):
    training_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    return training_dataloader, test_dataloader