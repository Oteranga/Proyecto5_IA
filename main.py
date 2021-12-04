import torch
import read_dataset as rd
import cnn as cnn
import plots as pl

lungs, training_size, validation_size, testing_size = rd.read_full_images()

training_set, validation_set, testing_set = rd.get_data_sets(
    lungs, training_size, validation_size)

training_loader, validation_loader, testing_loader = rd.get_data_loaders(
    training_set, validation_set, testing_set)

# CNN TRAINING #

num_classes = 4
learning_rate = 0.001
num_epochs = 15

""" model = cnn.CNN(num_classes, "batch").to(cnn.device)
loss_func = cnn.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
loss_list = cnn.train(model, optimizer, loss_func, num_epochs,training_loader)
pl.plot_error(loss_list,"error_batch","(Using Batch Normalization)") """

# TESTS #

def test_batch_norm():
    model = cnn.CNN(num_classes, "batch").to(cnn.device)
    loss_func = cnn.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    train_loss = cnn.training(model, optimizer, loss_func, num_epochs, training_loader, testing_loader, False)
    test_loss, accuracy = cnn.testing(model,loss_func,testing_loader)
    pl.plot_error(train_loss,test_loss,"t1_error_batch","(Using Batch Normalization)")

def test_dropout():
    model = cnn.CNN(num_classes, "dropout").to(cnn.device)
    loss_func = cnn.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    train_loss = cnn.training(model, optimizer, loss_func, num_epochs, training_loader, testing_loader, False)
    test_loss, accuracy = cnn.testing(model,loss_func,testing_loader)
    pl.plot_error(train_loss,test_loss,"t1_error_dropout","(Using Dropout)")

def test_batch_drop():
    model = cnn.CNN(num_classes, "both").to(cnn.device)
    loss_func = cnn.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    train_loss = cnn.training(model, optimizer, loss_func, num_epochs, training_loader,testing_loader,False)
    test_loss, accuracy = cnn.testing(model,loss_func,testing_loader)
    pl.plot_error(train_loss,test_loss,"t1_error_batch_drop","(Using Dropout and Batch Normalization)")

def test_early_stop():
    model = cnn.CNN(num_classes, "dropout").to(cnn.device)
    loss_func = cnn.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    train_loss = cnn.training(model, optimizer, loss_func, num_epochs, training_loader,testing_loader,True)
    test_loss, accuracy = cnn.testing(model,loss_func,testing_loader)
    pl.plot_error(train_loss,test_loss,"t1_error_early_stop","(Using Early Stopping)")

test_batch_norm()
