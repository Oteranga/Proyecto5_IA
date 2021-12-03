import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping

class CNN(nn.Module):
    def __init__(self, num_classes, type):
        super(CNN, self).__init__()
        if type == "batch":
            self.just_batch_norm()
        elif type == "dropout":
            self.just_dropout()
        else:
            self.batch_and_drop()
        self.fc = nn.Linear(7*7*32, num_classes)

    def just_batch_norm(self):
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=4))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=5))
        
    def just_dropout(self):
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=4))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=5))
    
    def batch_and_drop(self):
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=4))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=5))            

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


# CNN TRAINING FUNC #

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(model,optimizer,loss_func,num_epochs,training_loader, early_stop):
    print('Training CNN...')
    total_step = len(training_loader)
    loss_list = []
    time_list = []
    j = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(training_loader):
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            loss = loss_func(out,labels)
            
            optimizer.zero_grad() #zero the gradient
            loss.backward()
            optimizer.step() #updates
            
            loss_list.append(loss.item())
            time_list.append(j)
            j += 1
            
            if(i+1)%100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        
        
    print('Finished Training Trainset')
    return loss_list

def testing(model, loss_func, testing_loader, testing_dataset):
    print('Testing CNN...')
    model.eval()
    # for loss
    counter = 0
    loss_list = []
    # for accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testing_loader:
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            # for loss
            loss = loss_func(out,labels)            
            loss_list.append(loss.item())
            # for accuracy
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
        return loss_list