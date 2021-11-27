import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


# CNN TRAINING FUNC #

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(model,optimizer,loss_func,num_epochs,training_loader):
    total_step = len(training_loader)
    loss_list = []
    time_list = []
    j = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(training_loader):
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            loss = loss_func(out,labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_list.append(loss.item())
            time_list.append(j)
            j += 1
            
            if(i+1)%100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    print('Finished Training Trainset')
    return loss_list