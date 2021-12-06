import torch
import torch.nn as nn
#from ignite.engine import Engine, Events
#from ignite.handlers import EarlyStopping

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
trigger_times = 0

def training(model,optimizer,loss_func,num_epochs,training_loader, validation_loader, early_stop):
    print('Training CNN...')
    # for training
    total_step = len(training_loader)
    loss_training = []
    # for validation
    loss_validation = []
    # for early stopping
    last_val_loss = 100
    for epoch in range(num_epochs):
        current_train_loss = 0
        for i, (images, labels) in enumerate(training_loader):
            # for training
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            loss = loss_func(out,labels)
            optimizer.zero_grad() #zero the gradient
            loss.backward()
            optimizer.step() #updates
            
            current_train_loss += loss.item()
            
            if(i+1)%100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
        loss_training.append(current_train_loss/len(training_loader))
        
        current_val_loss = validation(model,loss_func,validation_loader)
        loss_validation.append(current_val_loss)

        if early_stop is True:
            last_val_loss = early_stopping(model,current_val_loss,last_val_loss)

    print('Finished Training CNN.')
    return loss_training, loss_validation

def early_stopping(model, current_val_loss, last_val_loss):
    patience = 5
    print('The current loss:', current_val_loss)
    if current_val_loss > last_val_loss:
        trigger_times += 1
        print('trigger times:', trigger_times)
        if trigger_times >= patience:
            print('Early stopping!\nStart to test process.')
            return model
    else:
        print('trigger times: 0')
        trigger_times = 0
    return current_val_loss

# CNN VALIDATION FUNC

def validation(model,loss_func,validation_loader):
    #model.eval()
    current_val_loss = 0
    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_func(outputs, labels)
            current_val_loss += loss.item()

    return current_val_loss / len(validation_loader)

# CNN TESTING FUNC #

def testing(model, testing_loader):
    print('Testing CNN...')
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testing_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    print('Finished Testing CNN.')