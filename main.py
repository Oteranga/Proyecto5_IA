import torch
import read_dataset as rd
import cnn as cnn
import plots as pl

lungs, training_size, validation_size, testing_size = rd.read_full_images()

training_set, validation_set, testing_set = rd.get_data_sets(
    lungs, training_size, validation_size)

training_loader, validation_loader, testing_loader = rd.get_data_loaders(
    training_set, validation_set, testing_set)

# EXPERIMENTS #

num_classes = 4
learning_rate = 0.001
num_epochs = 15

def experiment(exp_info):
    model = cnn.CNN(num_classes, exp_info[0]).to(cnn.device)
    loss_func = cnn.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    training_loss, validation_loss = cnn.training(
        model, optimizer, loss_func, num_epochs, training_loader, validation_loader, False)
    cnn.testing(model,loss_func,testing_loader)
    pl.plot_error(training_loss,validation_loss,exp_info[1],exp_info[2])

exp_batch_info = ["batch","t1_error_batch","(Using Batch Normalization)"]
exp_dropout_info = ["dropout","t1_error_dropout","(Using Dropout)"]
exp_batch_drop_info = ["both","t1_error_batch_drop","(Using Dropout and Batch Normalization)"]
exp_early_stop_info = ["dropout","t1_error_early_stop","(Using Early Stopping)"]

experiment(exp_batch_info)
