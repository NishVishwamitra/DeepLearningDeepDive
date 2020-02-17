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
import math
import random

# define non linear function
def non_linear_function(x):
  return math.sin(5 * math.pi * x) / (5 * math.pi * x)

# prepare dataset
dataset = []
for i in range(1, 2001):
  r = random.random()
  dataset.append((r, non_linear_function(r)))

# split into train and test
train_dataset = dataset[:int(len(dataset) * .8)]
test_dataset = dataset[int(len(dataset) * .8):]

# use pytorch dataloaders for convenience and batching
train_loader = DataLoader(train_dataset, batch_size = 4, shuffle = True, num_workers = 4)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = 4)

# deep model
class DeepMLP(nn.Module):
  def __init__(self):
    super(DeepMLP, self).__init__()

    self.mlp = nn.Sequential(
      nn.Linear(1, 128),
      nn.Tanh(),
      nn.Linear(128, 64),
      nn.Tanh(),
      nn.Linear(64, 32),
      nn.Tanh(),
      nn.Linear(32, 16),
      nn.Tanh(),
      nn.Linear(16, 1),
      nn.Tanh()) 
    
  def forward(self, x):
    return self.mlp(x)

# loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
model = DeepMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# train
def train(model):
  model.train()
  avg_loss = []
  for i in train_loader:
    x, y = i    
    x, y = x.to(device, dtype = torch.float32).unsqueeze(1), y.to(device, dtype = torch.float32).unsqueeze(1)
    #x, y = x.to(device, dtype = torch.float32), y.to(device, dtype = torch.float32)
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
      x, y = x.to(device, dtype = torch.float32), y.to(device, dtype = torch.float32)
      y_hat = model(x)
      loss = criterion(y_hat, y)
      avg_loss.append(torch.mean(loss).item())

      if abs(y_hat.item() ** 2 - y.item() ** 2) < 0.01:
        correct += 1
    
      total += 1

  return sum(avg_loss) / len(avg_loss), (correct / total) * 100
    
epochs = 15

for epoch in range(epochs):
  train_loss = train(model)
  test_loss, acc = test(model)
  print('Epoch:', epoch + 1, ', train loss:', train_loss, ', test loss:', test_loss, ', accuracy:', acc)
