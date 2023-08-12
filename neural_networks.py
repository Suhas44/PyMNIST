import torch
from torch import nn, optim


class MultilayerPerceptron(nn.Module):

    def __init__(self, device):
        super(MultilayerPerceptron, self).__init__()
        self.flatten = nn.Flatten()  
        self.MLP_sequential = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10) 
        )
        self.loss_function = nn.CrossEntropyLoss()
        self.device = device
        self = self.to(device)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.MLP_sequential(x)
        return logits