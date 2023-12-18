import torch
import torch.nn as nn

# 入力はMNIST
class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(784, 1024, bias = False)
		self.fc2 = nn.Linear(1024, 10, bias = False)
	def forward(self, x):
		x = x.reshape(-1, 784)
		x = self.fc1(x)
		x = torch.relu(x)
		x = self.fc2(x)
		return x

class QuantizedModel(Model):
	def __init__(self, shiftM):
		super().__init__()
		self.shiftM = shiftM
	def forward(self, x):
		x = x.reshape(-1, 784)
		x = self.fc1(x)
		x = torch.relu(x)
		# 再量子化
		x = (x.int() >> self.shiftM).float()
		x = self.fc2(x)
		return x

