import torch
import read_dataset as rd
import cnn as cnn
import plots as pl

lungs, training_size, validation_size, testing_size = rd.read_partial_images(0.1)

training_set, validation_set, testing_set = rd.get_data_sets(
    lungs, training_size, validation_size)

training_loader, validation_loader, testing_loader = rd.get_data_loaders(
    training_set, validation_set, testing_set)

""" print(len(training_loader))
for i in range(len(training_loader)):
    img,label = training_set[i]
    print(img.shape) """

# CNN TRAINING #

num_classes = 4
learning_rate = 0.001
num_epochs = 15

model = cnn.CNN(num_classes, "batch").to(cnn.device)
loss_func = cnn.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
loss_list = cnn.train(model, optimizer, loss_func, num_epochs,training_loader)
pl.plot_error(loss_list,"error_batch","(Using Batch Normalization)")

# TESTS #

def test_batch_norm():
    model = cnn.CNN(num_classes, "batch").to(cnn.device)
    loss_func = cnn.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    loss_list = cnn.train(model, optimizer, loss_func, num_epochs, training_loader)
    pl.plot_error(loss_list,"t1_error_batch","(Using Batch Normalization)")

def test_dropout():
    model = cnn.CNN(num_classes, "dropout").to(cnn.device)
    loss_func = cnn.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    loss_list = cnn.train(model, optimizer, loss_func, num_epochs, training_loader)
    pl.plot_error(loss_list,"t1_error_dropout","(Using Dropout)")

def test_early_stop():
    model = cnn.CNN(num_classes, "dropout").to(cnn.device)
    loss_func = cnn.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    loss_list = cnn.train(model, optimizer, loss_func, num_epochs, training_loader)
    pl.plot_error(loss_list,"t1_error_early_stop","(Using Early Stopping)")
