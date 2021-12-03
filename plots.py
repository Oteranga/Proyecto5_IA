import torch
import torch.nn as nn
import cnn as cnn
import matplotlib.pyplot as plt
import numpy as np


def plot_error(loss_list, img_name, title):
    plt.plot(np.array(loss_list), 'r')
    plt.title("Error vs Epoch" + title)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.savefig("plots/" + img_name)

def get_accuracy(loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(cnn.device)
            labels = labels.to(cnn.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))