import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim
import model
import os

# データのダウンロードとデータセットの作成
train_dataset = datasets.MNIST(root='../data', train=True, transform=transforms.ToTensor(), download = True)
test_dataset = datasets.MNIST(root='../data', train=False, transform=transforms.ToTensor(), download = True)
batch_size = 64
train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

weights = torch.load("weight.oth")
model1 = model.Model()
model1.load.state_dict(weights)
fmodel = model.FoldedModel()

w_conv1 = model1.conv1.weight.data
b_conv1 = model1.conv1.bias.data
w_bn1 = model1.bn1.weight.data
b_bn1 = model1.bn1.bias.data
m_bn1 = model1.bn1.running_mean.data
v_bn1 = model1.bn1.runnig_var.data1

w_conv1_folded = w_conv1 * w_bn1 / v_bn1.sqrt()
b_conv1_folded = 

w_conv2 = model1.conv2.weight.data
b_conv2 = model1.conv2.bias.data
w_bn2 = model1.bn2.weight.data
b_bn2 = model1.bn2.bias.data
m_bn2 = model1.bn2.running_mean.data
v_bn2 = model1.bn2.runnig_var.data

