import torch
import torch.nn as nn

# 入力はMNIST
class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 16, kernel_size = 5, padding = 0)
		self.bn1 = nn.BatchNorm2d(16)
		self.maxpool1 = nn.MaxPool2d(2, stride = 2)
		self.conv2 = nn.Conv2d(16, 32, kernel_size = 5, padding = 0)
		self.bn2 = nn.BatchNorm2d(32)
		self.maxpool2 = nn.MaxPool2d(2, stride = 2)
		self.fc1 = nn.Linear(32*4*4, 1024, bias = True)
		self.fc2 = nn.Linear(1024, 10, bias = True)
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = torch.relu(x)
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = torch.relu(x)
		x = self.maxpool2(x)
		x = x.reshape(-1, 32*4*4)
		x = self.fc1(x)
		x = torch.relu(x)
		x = self.fc2(x)
		return x

class FoldedModel(Model):
	def __init__(self):
		super().__init__()
	def forward(self, x):
		x = self.conv1(x)
		x = torch.relu(x)
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = torch.relu(x)
		x = self.maxpool2(x)
		x = x.reshape(-1, 32*4*4)
		x = self.fc1(x)
		x = torch.relu(x)
		x = self.fc2(x)
		return x

class QuantizedModel(Model):
	def __init__(self, shiftM1, shiftM2, shiftM3):
		super().__init__()
		self.shiftM1 = shiftM1
		self.shiftM2 = shiftM2
		self.shiftM3 = shiftM3
	def forward(self, x):
		x = self.conv1(x)
		x = torch.relu(x)
		x = (x.int() >> self.shiftM1).float()
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = torch.relu(x)
		x = (x.int() >> self.shiftM2).float()
		x = self.maxpool2(x)
		x = x.reshape(-1, 32*4*4)
		x = self.fc1(x)
		x = torch.relu(x)
		x = (x.int() >> self.shiftM3).float()
		x = self.fc2(x)
		return x

