import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim
import model

# データのダウンロードとデータセットの作成
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download = True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download = True)
batch_size = 64
train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

# トレーニング
model = model.Model()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

def train_step(x, t):
	model.train()
	x = (x * 255).round() # 入力の量子化
	output = model(x)
	loss = criterion(output, t)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	return loss

epochs = 5
for e in range(epochs):
	total_loss = 0
	for (x, t) in train_dataloader:
		loss = train_step(x, t)
		total_loss += loss
	print(e, ":", total_loss)

# パラメータの保存
torch.save(model.state_dict(), "weight.pth")
