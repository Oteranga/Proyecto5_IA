import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math
import read_dataset as rd

num_classes = 10
learning_rate = 0.001
num_epochs = 20

class CNN(nn.Module):
    def __init__(self, num_classes = 4):
        super(CNN, self).__init__()

        #Input layer
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2),nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2))

        #Hidden layer
        """ self.layerH1 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2))
        self.layerH2 = nn.Sequential(
        nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=2),nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)) """

        #Output layer
        self.layer2 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2))
        #nn.AdaptiveMaxPool2d(output_size = 7*7*32))

        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        """ out = self.layerH1(out)
        out = self.layerH2(out) """
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

network = CNN()
lungs, train_size, validate_size, test_size = rd.read_images()
train_x = rd.convert(lungs[0:train_size])
val_x = rd.convert(lungs[train_size + 1: train_size + validate_size])
test_x = rd.convert(lungs[train_size + validate_size: -1])

training, validation, testing = rd.data_sets(lungs, train_size, validate_size, test_size)

""" train_x = rd.get_torch(train_x)
val_x = rd.get_torch(val_x)
test_x = rd.get_torch(test_x) """
img = train_x[10].unsqueeze(0)

network.forward(img)