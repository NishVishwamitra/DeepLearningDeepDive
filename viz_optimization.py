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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
class MNISTDNN(nn.Module):
  def __init__(self):
    super(MNISTDNN, self).__init__()

    self.dnn = nn.Sequential(
      nn.Linear(28 * 28, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, 10),
      nn.Sigmoid()) 
    
  def forward(self, x):
    x = x.view(x.size(0), 28 * 28)
    x = self.dnn(x)
    return x

# loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
model = MNISTDNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)

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
    
epochs = 80
pca = PCA(n_components = 2)
all_weights = []

for epoch in range(epochs):
  train_loss = train(model)
  test_loss, acc = test(model)
  print('Epoch:', epoch + 1, ', train loss:', train_loss, ', test loss:', test_loss, ', accuracy:', acc)

  if epoch % 2 == 0:
    l0_w = model.dnn[0].weight
    l1_w = model.dnn[2].weight
    l2_w = model.dnn[4].weight
    l3_w = model.dnn[6].weight
    
    # flatten each layer weight and concat
    l0_w_flat = torch.flatten(l0_w)
    l1_w_flat = torch.flatten(l1_w)
    l2_w_flat = torch.flatten(l2_w)
    l3_w_flat = torch.flatten(l3_w)

    combined_weights = torch.cat((l0_w_flat, l1_w_flat, l2_w_flat, l3_w_flat))
    all_weights.append(combined_weights.detach().cpu().numpy())

all_weights = np.array(all_weights)
principal_components = pca.fit_transform(all_weights)

plt.scatter(principal_components[:,0], principal_components[:,1])

plt.savefig('weights_plot.png')
