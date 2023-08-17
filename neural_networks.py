import torch
from torch import nn, optim


class MultilayerPerceptron(nn.Module):

    def __init__(self, device):
        super(MultilayerPerceptron, self).__init__()
        self.device = device

        self.flatten = nn.Flatten()  
        self.MLP_sequential = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10) 
        )
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.MLP_sequential(x)
        return logits

class ConvolutionalNeuralNetwork(nn.Module):

    def __init__(self, device):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.device = device
        self.loss_function = nn.CrossEntropyLoss()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output