from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
from get_data import get_dataloaders
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor()
)