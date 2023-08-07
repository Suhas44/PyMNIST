from torch import optim
from torch.autograd import Variable
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
from get_data import get_training_dataloader
from nn_models import MultilayerPerceptron

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_dataloader = get_training_dataloader()


n_epochs = 10

def train(n_epochs, model, training_data, optimizer):
    model.train()

    n_steps = len(training_data)

    for epoch in range(n_epochs):

        for index, (images, labels) in enumerate(training_data):
            img_batch = Variable(images)
            label_batch = Variable(labels)

            output = model(img_batch)
            loss = model.loss_function(output, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (index + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, n_epochs, index + 1, n_steps, loss.item()))

MLP = MultilayerPerceptron()
optimizer = optim.Adam(MLP.parameters(), lr=0.01)
train(n_epochs=n_epochs, model=MLP, training_data=training_dataloader, optimizer=optimizer)
torch.save(MLP, 'multilayer_perceptron.pth')