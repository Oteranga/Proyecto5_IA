import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import glob
import math
import numpy as np
import random

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
    
    train_size = math.ceil(len(lungs) * 0.7)
    validate_size = math.floor(len(lungs) * 0.2)
    test_size = math.floor(len(lungs) * 0.1)

    random.shuffle(lungs)
    return lungs, train_size, validate_size, test_size

def data_sets(lungs, train_size, validate_size, test_size):

    training = torch.utils.data.DataLoader(dataset = lungs, batch_size = train_size, shuffle = False)
    validation = torch.utils.data.DataLoader(dataset = lungs, batch_size = validate_size, shuffle = False)
    testing = torch.utils.data.DataLoader(dataset = lungs, batch_size = test_size, shuffle = False)

    return training, validation, testing

def convert(data):
    data_x = []

    for img in data:
        data_x.append(img[0])

    return data_x

def get_torch(data_x):
    #set_x = np.array(data_x)
    #set_x = set_x.reshape(size, 1, 28, 28)
    #set_x = torch.from_numpy(set_x)
    for val in data_x:
        val = val.unsqueeze(0)

    return data_x

