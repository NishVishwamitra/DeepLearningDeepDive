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


def f(x):
  return math.sin(5 * math.pi * x) / (5 * math.pi * x)

# prepare dataset
dataset = []
for i in range(1, 2001):
  r = random.random()
  dataset.append((r, f(r)))

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

    # using shallow network to simplify minimizing the grad norm for this experiment
    self.mlp = nn.Sequential(
      nn.Linear(1, 64),
      nn.Tanh(),
      nn.Linear(64, 64), # weight of this layer is square, easy for hessian
      nn.Tanh(),
      nn.Linear(64, 1),
      nn.Tanh()) 
    
  def forward(self, x):
    return self.mlp(x)

# loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
model = DeepMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

def hv(loss, model, v):
  grad = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
  Hv = torch.autograd.grad(grad, model.parameters(), grad_outputs=v, retain_graph=True)
  return Hv, grad

# train
def train(model, epoch):
  model.train()
  avg_loss = []
  for i in train_loader:
    x, y = i    
    x, y = x.to(device, dtype = torch.float32).unsqueeze(1), y.to(device, dtype = torch.float32).unsqueeze(1)
    y_hat = model(x)
    
    # let the model train first, then start minimizing gradnorm to make to close to zero
    if epoch < 40:
      loss = criterion(y_hat, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step() 
      avg_loss.append(torch.mean(loss).item())
      grad_norm = (((model.mlp[0].weight.grad) ** 2).sum() + 
        ((model.mlp[2].weight.grad) ** 2).sum() + ((model.mlp[4].weight.grad) ** 2).sum()) ** 0.5
    else:
  
      loss = torch.tensor(0.).to(device)
      for p in model.parameters():
        loss += criterion(p, torch.zeros(p.size()).to(device))

      #loss = criterion(grad_norm, torch.zeros(y.size()).to(device)) 
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      avg_loss.append(torch.mean(loss).item())
      grad_norm = (((model.mlp[0].weight.grad) ** 2).sum() + 
        ((model.mlp[2].weight.grad) ** 2).sum() + ((model.mlp[4].weight.grad) ** 2).sum()) ** 0.5
    
  print(grad_norm)

  if epoch == epochs - 1 or epoch == epochs - 2:
    for p in model.parameters():
      print(p)

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
    
epochs = 50

for epoch in range(epochs):
  train_loss = train(model, epoch)
  test_loss, acc = test(model)
  print('Epoch:', epoch + 1, ', train loss:', train_loss, ', test loss:', test_loss, ', accuracy:', acc)
