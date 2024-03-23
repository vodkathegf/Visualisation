import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
from prettytable import PrettyTable

matplotlib.use('TkAgg')


class FeedForward(nn.Module):
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        super(FeedForward, self).__init__()
        self.hidden = nn.Linear(input_neurons, hidden_neurons)
        self.out = nn.Linear(hidden_neurons, output_neurons)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.out(x)
        return x


class AccuracyCalculator:
    def __init__(self):
        self.accuracies = []
        self.epoch_accuracies = []

    def calculate_accuracy(self, predicted_y, target_y, epoch):
        accuracy = (predicted_y == target_y).sum() / target_y.size
        self.accuracies.append(accuracy)
        if (epoch + 1) % 100 == 0:
            self.epoch_accuracies.append(accuracy)
        return accuracy

    def get_epoch_accuracies(self):
        return self.epoch_accuracies


class VisualizationHandler:
    def __init__(self):
        plt.ion()

    def plot_data(self, x, predicted_y, accuracy):
        plt.clf()
        plt.scatter(x[:, 0], x[:, 1], s=40, c=predicted_y, linewidths=0)
        centroid_x = np.mean(x[:, 0])
        centroid_y = np.mean(x[:, 1])
        plt.text(centroid_x, np.max(
            x[:, 1]) + 0.1 or centroid_y, "accuracy equals to {:.2f}".format(accuracy), fontdict={'size': 14})
        plt.pause(0.1)

    def plot_accuracy(self, epochs, accuracies):
        plt.figure()
        plt.plot(epochs, accuracies, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.grid(True)
        # I cannot solve the closing plt elements  problem
        # so I make it wait 100s
        plt.pause(100)



class ModelTrainer:
    def __init__(self, model, optimizer, loss_function, accuracy_calculator, visualization_handler):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.accuracy_calculator = accuracy_calculator
        self.visualization_handler = visualization_handler

    def train(self, x, y, epochs):
        for epoch in range(epochs):
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                # Calculate accuracy
                max_value, prediction = torch.max(out, 1)
                predicted_y = prediction.data.numpy()
                target_y = y.data.numpy()
                accuracy = self.accuracy_calculator.calculate_accuracy(
                    predicted_y, target_y, epoch)
                self.visualization_handler.plot_data(
                    x.data.numpy(), predicted_y, accuracy)

        epochs_range = range(0, epochs, 100)
        self.visualization_handler.plot_accuracy(
            epochs_range, self.accuracy_calculator.accuracies)
        plt.show()



def main():
    x, y = sklearn.datasets.make_moons(200, noise=0.18)
    x = torch.FloatTensor(x)
    y = torch.LongTensor(y)

    model = FeedForward(input_neurons=2, hidden_neurons=50, output_neurons=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()
    accuracy_calculator = AccuracyCalculator()
    visualization_handler = VisualizationHandler()

    trainer = ModelTrainer(model, optimizer, loss_function,
                           accuracy_calculator, visualization_handler)
    trainer.train(x, y, epochs=1000)


if __name__ == "__main__":
    main()
