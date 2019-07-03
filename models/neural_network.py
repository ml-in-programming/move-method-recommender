import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()

        self.__linear1 = nn.Linear(input_size, 128)
        self.__linear2 = nn.Linear(128, 32)
        self.__linear3 = nn.Linear(32, 1)
        self.__activation = nn.Sigmoid()

    def forward(self, x):
        x = self.__linear1(x)
        x = self.__linear2(x)
        x = self.__linear3(x)
        return self.__activation(x)


class Trainer:
    def __init__(self, net, loss, optimizer, writer=None):
        self.__net = net
        self.__loss = loss
        self.__optimizer = optimizer
        self.__writer = writer

    def train(self, dataset, batch_size, epochs_num):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.__net.train(True)

        for epoch in range(epochs_num):
            for inputs, labels in data_loader:
                self.__optimizer.zero_grad()
                outputs = self.__net(inputs)

                loss_value = self.__loss(outputs, labels.float())
                loss_value.backward()
                self.__optimizer.step()

            # training_loss_after_epoch = self.__average_loss(data_loader)

            # if self.__writer is not None:
            #    self.__writer.add_scalar('Training loss', training_loss_after_epoch, epoch + 1)

        self.__net.train(False)


class NeuralNetworkModel:
    def __init__(self, model):
        self.__model = model

    def __call__(self, points):
        with torch.no_grad():
            return self.__model(torch.tensor(points).float()).numpy()


def create_neural_network(dataset):
    dataset = TensorDataset(torch.tensor(dataset.X).float(), torch.tensor(dataset.y))

    model = NeuralNetwork(dataset[0][0].shape[0])
    trainer = Trainer(model, nn.BCELoss(reduction='sum'), optim.SGD(model.parameters(), lr=0.005))
    trainer.train(dataset, 16, 50)

    return NeuralNetworkModel(model)
