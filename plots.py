import torch
import torch.nn as nn
import cnn as cnn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_error(train_list, test_list, img_name, title):
    fig, ax = plt.subplots()
    ax.plot(train_list, 'r', label = 'train loss')
    ax.plot(test_list, 'b', label = 'test loss')
    #df = pd.DataFrame({"epochs": 30, "train": train_list, "test":test_list})
    #df.plot(x = "epochs")
    plt.title("Error vs Epoch" + title)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    ax.legend()
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