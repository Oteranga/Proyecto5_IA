import torch
from torchvision import transforms
from PIL import Image, ImageOps
import glob
import math
import random
import os

batch_size = 64

def read_full_images():
    lungs = []
    directories = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
    for word in directories:
        path = "./Dataset/" + word + "/*"
        for img in glob.iglob(path):
            conver_tensor = transforms.ToTensor()
            new_img = conver_tensor(ImageOps.grayscale(Image.open(img)))
            if word == "COVID":
                lungs.append((new_img, 0))
            elif word == "Lung_Opacity":
                lungs.append((new_img, 1))
            elif word == "Normal":
                lungs.append((new_img, 2))
            else:
                lungs.append((new_img, 3))
    
    training_size = math.ceil(len(lungs) * 0.7)
    validation_size = math.floor(len(lungs) * 0.2)
    testing_size = math.floor(len(lungs) * 0.1)

    random.shuffle(lungs)
    return lungs, training_size, validation_size, testing_size


def read_partial_images(percentage):
    path_, dirs, files = next(os.walk("./Dataset/Normal"))
    file_count = len(files)
    img_quant = percentage*file_count

    lungs = []
    directories = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
    for word in directories:
        path = "./Dataset/" + word + "/*"
        i = 0
        for img in glob.iglob(path):
            if i<img_quant:
                conver_tensor = transforms.ToTensor()
                new_img = conver_tensor(ImageOps.grayscale(Image.open(img)))
                if word == "COVID":
                    lungs.append((new_img, 0))
                elif word == "Lung_Opacity":
                    lungs.append((new_img, 1))
                elif word == "Normal":
                    lungs.append((new_img, 2))
                else:
                    lungs.append((new_img, 3))
                i += 1
            else:
                break
    
    training_size = math.ceil(len(lungs) * 0.7)
    validation_size = math.floor(len(lungs) * 0.2)
    testing_size = math.floor(len(lungs) * 0.1)

    random.shuffle(lungs)
    return lungs, training_size, validation_size, testing_size

def get_data_sets(lungs, train_size, validation_size):
    training_set = lungs[0:train_size]
    validation_set = lungs[train_size+1:train_size+validation_size]
    testing_set = lungs[train_size+validation_size+1:-1]
    return training_set, validation_set, testing_set


def get_data_loaders(training_set, validation_set, testing_set):
    training_loader = torch.utils.data.DataLoader(
        dataset = training_set, batch_size = batch_size, shuffle = True)
    validation_loader = torch.utils.data.DataLoader(
        dataset = validation_set, batch_size = batch_size, shuffle = False)
    testing_loader = torch.utils.data.DataLoader(
        dataset = testing_set, batch_size = batch_size, shuffle = False)
    return training_loader, validation_loader, testing_loader

