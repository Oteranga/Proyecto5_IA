import torch
from torchvision import transforms
from PIL import Image
import glob
import math
import random

batch_size = 64

def read_images():
    covid = []
    lung_opacity = []
    normal = []
    pneumonia = []
    lungs = []
    directories = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

    for word in directories:
        path = "./Dataset/" + word + "/*"
        for img in glob.iglob(path):
            conver_tensor = transforms.ToTensor()
            new_img = conver_tensor(Image.open(img))
            if word == "COVID":
                covid.append(new_img)
                lungs.append((new_img, 0))
            elif word == "Lung_Opacity":
                lung_opacity.append(new_img)
                lungs.append((new_img, 1))
            elif word == "Normal":
                normal.append(new_img)
                lungs.append((new_img, 2))
            else:
                pneumonia.append(new_img)
                lungs.append((new_img, 3))
    
    training_size = math.ceil(len(lungs) * 0.7)
    validation_size = math.floor(len(lungs) * 0.2)
    testing_size = math.floor(len(lungs) * 0.1)

    random.shuffle(lungs)
    return lungs, training_size, validation_size, testing_size

def get_data_sets(lungs, train_size, validation_size, test_size):
    training_set = lungs[0:train_size]
    validation_set = lungs[train_size+1:train_size+validation_size]
    testing_set = lungs[train_size+validation_size+1:-1]
    return training_set, validation_set, testing_set


def get_data_loaders(training_set, validation_set, testing_set):
    training_loader = torch.utils.data.DataLoader(
        dataset = training_set, batch_size = batch_size, shuffle = False)
    validation_loader = torch.utils.data.DataLoader(
        dataset = validation_set, batch_size = batch_size, shuffle = False)
    testing_loader = torch.utils.data.DataLoader(
        dataset = testing_set, batch_size = batch_size, shuffle = False)
    return training_loader, validation_loader, testing_loader

