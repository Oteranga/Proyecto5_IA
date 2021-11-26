import CNN as CNN
import torch
import torch.nn as nn
import torchvision

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(model, optimizer, loss_fn, epochs, train_loader):
  loss_vals = []
  running_loss = 0.0
  # train the model
  total_step = len(train_loader)

  list_loss= []
  list_time = []
  j=0

  for epoch in range(epochs):
    for i, (images) in enumerate(train_loader):
      images = images.to(device)
      # forward 
      output = model(images)
      loss   = loss_fn(output)
      # change the params
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      list_loss.append(loss.item())
      list_time.append(j)
      j+=1
              
      if (i+1) % 100 == 0:
              print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, total_step, loss.item()))
              
  print('Finished Training Trainset')
  return list_loss

model = CNN().to(device)
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
loss_method = nn.CrossEntropyLoss()
epochs = 50
