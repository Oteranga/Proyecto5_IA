import torch
import torch.nn as nn
import cnn as cnn
import matplotlib.pyplot as plt
import numpy as np


def plot_error(loss_list):
    plt.plot(np.array(loss_list), 'r')

def get_accuracy(testing_loader, model):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testing_loader:
            images = images.to(cnn.device)
            labels = labels.to(cnn.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))