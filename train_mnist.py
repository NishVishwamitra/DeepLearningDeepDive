import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
from torchvision import transforms, utils
import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision
import math
import random

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/scratch1/nvishwa', train = True, download = True,
  transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
  batch_size = 4, shuffle = True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/scratch1/nvishwa', train = False, download = True,
  transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
  batch_size = 1, shuffle = False)

# deep model
class MNISTCNN(nn.Module):
  def __init__(self):
    super(MNISTCNN, self).__init__()

    self.cnn = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size = 2, stride = 2),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size = 4, stride = 2),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.ReLU()) 

    self.lin = nn.Sequential(
      nn.Linear(64, 10),
      nn.Sigmoid())
    
  def forward(self, x):
    x = self.cnn(x)
    x = x.view(x.size(0), -1)
    x = self.lin(x)
    return x

# loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
model = MNISTCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

# train
def train(model):
  model.train()
  avg_loss = []
  for i in train_loader:
    x, y = i    
    x, y = x.to(device), y.to(device)
    y_hat = model(x)
    loss = criterion(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 
    avg_loss.append(torch.mean(loss).item())
  return sum(avg_loss) / len(avg_loss)

# test
def test(model):
  model.eval()
  avg_loss = []
  avg_acc = []
  correct = 0.
  total = 0.
  with torch.no_grad():
    for i in test_loader:
      x, y = i
      x, y = x.to(device), y.to(device)
      y_hat = model(x)
      loss = criterion(y_hat, y)
      avg_loss.append(torch.mean(loss).item())

      if torch.argmax(y_hat) == y:
        correct += 1
    
      total += 1

  return sum(avg_loss) / len(avg_loss), (correct / total) * 100
    
epochs = 10

for epoch in range(epochs):
  train_loss = train(model)
  test_loss, acc = test(model)
  print('Epoch:', epoch + 1, ', train loss:', train_loss, ', test loss:', test_loss, ', accuracy:', acc)
