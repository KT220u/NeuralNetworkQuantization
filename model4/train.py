import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

batch_size = 128
train_dataloader = DataLoader(
	datasets.CIFAR10(
		"../data",
		train = True,
		download = True,
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(
				[0.5, 0.5, 0.5],
				[0.5, 0.5, 0.5]
			)
		])
	), 
	batch_size = batch_size,
	shuffle = True
)

test_dataloader = DataLoader(
	datasets.CIFAR10(
		"../data",
		train = False,
		download = True,
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(
				[0.5, 0.5, 0.5],
				[0.5, 0.5, 0.5]
			)
		])
	),
	batch_size = batch_size,
	shuffle = False
)

# train
import model
import os

def train_step(x, t):
	model1.train()
	output = model1(x)
	loss = criterion(output, t)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	return loss

model1 = model.Model()
if os.path.exists("weight.pth"):
	model1.load_state_dict(torch.load("weight.pth"))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model1.parameters(), lr = 0.1)

epochs = 3
for e in range(epochs):
	total_loss = 0
	for (x, t) in train_dataloader:
		loss = train_step(x, t)
		total_loss += loss
	print("epoch :", e, ", total loss :", total_loss)

# パラメータの保存
weights = model1.state_dict()
torch.save(weights, "weight.pth")

