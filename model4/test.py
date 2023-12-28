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


import model

model1 = model.Model()
model1.load_state_dict(torch.load("weight.pth"))

total = 0
correct = 0
for (x, t) in test_dataloader:
	model1.eval()
	output = model1(x)
	_, predicted = torch.max(output, 1)
	total += output.shape[0]
	correct += (predicted == t).sum()
print("correct rate : ", (correct / total).item())

