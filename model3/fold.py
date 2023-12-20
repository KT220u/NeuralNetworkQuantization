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

weights = torch.load("weight.pth")
model1 = model.Model()
model1.load_state_dict(weights)
fmodel = model.FoldedModel()

w_conv1 = model1.conv1.weight.data
b_conv1 = model1.conv1.bias.data
w_bn1 = model1.bn1.weight.data
b_bn1 = model1.bn1.bias.data
m_bn1 = model1.bn1.running_mean.data
v_bn1 = model1.bn1.running_var.data

w_conv1_folded = w_conv1.clone()
for i in range(w_conv1.shape[0]):
	w_conv1_folded[i] = w_conv1[i] * w_bn1[i] / v_bn1[i].sqrt()
b_conv1_folded = (b_conv1 - m_bn1) * w_bn1 / v_bn1.sqrt() + b_bn1

fmodel.conv1.weight.data = w_conv1_folded
fmodel.conv1.bias.data = b_conv1_folded

w_conv2 = model1.conv2.weight.data
b_conv2 = model1.conv2.bias.data
w_bn2 = model1.bn2.weight.data
b_bn2 = model1.bn2.bias.data
m_bn2 = model1.bn2.running_mean.data
v_bn2 = model1.bn2.running_var.data

w_conv2_folded = w_conv2.clone()
for i in range(w_conv2.shape[0]):
	w_conv2_folded[i] = w_conv2[i] * w_bn2[i] / v_bn2[i].sqrt()
b_conv2_folded = (b_conv2 - m_bn2) * w_bn2 / v_bn2.sqrt() + b_bn2

fmodel.conv2.weight.data = w_conv2_folded
fmodel.conv2.bias.data = b_conv2_folded

fmodel.fc1.weight.data = model1.fc1.weight.data
fmodel.fc1.bias.data = model1.fc1.bias.data
fmodel.fc2.weight.data = model1.fc2.weight.data
fmodel.fc2.bias.data = model1.fc2.bias.data

correct = 0
total = 0
for (x, t) in test_dataloader:
	fmodel.eval()
	output = fmodel(x)
	_, predicted = torch.max(output, 1)
	total += output.shape[0]
	correct += (predicted == t).sum()
print("correct rate : ", (correct / total).item())

# 出力の確認
for (x, t) in test_dataloader:
		x = fmodel.conv1(x)
		x = torch.relu(x)
		x = fmodel.maxpool1(x)
		x = fmodel.conv2(x)
		x = torch.relu(x)
		x = fmodel.maxpool2(x)
		x = x.reshape(-1, 32*4*4)
		x = fmodel.fc1(x)
		x = torch.relu(x)
		x = fmodel.fc2(x)
		break
