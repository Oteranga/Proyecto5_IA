import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import glob
import math
import numpy as np

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
                lungs.append(tuple(new_img, 0))
            elif word == "Lung_Opacity":
                lung_opacity.append(new_img)
                lungs.append(tuple(new_img, 1))
            elif word == "Normal":
                normal.append(new_img)
                lungs.append(tuple(new_img, 2))
            else:
                pneumonia.append(new_img)
                lungs.append(tuple(new_img, 3))
    
    return lungs

def data_sets(lungs):
    train_size = math.ceil(len(lungs) * 0.7)
    validate_size = math.floor(len(lungs) * 0.2)
    test_size = math.floor(len(lungs) * 0.1)

    training = torch.utils.data.Dataloader(dataset = lungs, batch_size = train_size, shuffle = True)
    validation = torch.utils.data.Dataloader(dataset = lungs, batch_size = validate_size, shuffle = True)
    testing = torch.utils.data.Dataloader(dataset = lungs, batch_size = test_size, shuffle = True)

    return training, validation, testing

def convert(data):
    data_x = []
    data_y = []

    for img in data:
        data_x.append(img[0])
        data_y.append(img[1])

    return data_x, data_y

def get_torch(data_x, data_y, size):
    set_x = np.array(data_x)
    set_y = np.array(data_y)
    set_x = set_x.reshape(size, 1, 28, 28)
    set_x = torch.from_numpy(set_x)
    set_y = torch.from_numpy(set_y)

    return set_x, set_y

