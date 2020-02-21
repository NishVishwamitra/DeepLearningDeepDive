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

# Newton method for multiple dimensions explained very well here:
# https://medium.com/autonomous-agents/how-to-tame-the-valley-hessian-free-hacks-for-optimizing-large-neuralnetworks-5044c50f4b55

# Use of Hessian in monitoring optimization is: If the Hessian is positive definite (meaning all its eigenvalues are positive), then the function is a convex function.
#https://rohanvarma.me/Optimization/

def f(x):
  return math.sin(5 * math.pi * x) / (5 * math.pi * x)

def f_second(x):
  return -((225 * x ** 2 - 2) * (torch.sin(15 * x)) + (30 * x * torch.cos(15 * x))) / (15 * x ** 3)

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
optimizer_quasi_newton = torch.optim.LBFGS(model.parameters(), lr = 0.1)

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
      grad_norm = (((model.mlp[0].weight.grad) ** 2).sum() + ((model.mlp[2].weight.grad) ** 2).sum() + ((model.mlp[4].weight.grad) ** 2).sum()) ** 0.5
    else:
  
      def closure():
        if torch.is_grad_enabled():
          optimizer_quasi_newton.zero_grad()
          y_hat = model(x)
          loss = criterion(y_hat, y)
          if loss.requires_grad:
            loss.backward()
          return loss
            
      optimizer_quasi_newton.step(closure)
       
      grad_norm = (((model.mlp[0].weight.grad) ** 2).sum() + ((model.mlp[2].weight.grad) ** 2).sum() + ((model.mlp[4].weight.grad) ** 2).sum()) ** 0.5

      avg_loss.append(grad_norm)
  
  Hessian = f_second(model.mlp[2].weight)
  #print(Hessian.size())
  w, v = np.linalg.eig(Hessian.cpu().detach().numpy())
  min_ratio = w[w > 0]   
  print(min_ratio.shape)   

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
