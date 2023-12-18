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

# トレーニング
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

epochs = 10
for e in range(epochs):
	total_loss = 0
	for (x, t) in train_dataloader:
		loss = train_step(x, t)
		total_loss += loss
	print("epoch :", e, ", total loss :", total_loss)

# パラメータの保存
weights = model1.state_dict()
torch.save(weights, "weight.pth")

