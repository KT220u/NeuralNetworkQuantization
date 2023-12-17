import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim
import model
import sys

train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download = True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download = True)
batch_size = 64
train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

weights = torch.load("weight.pth")
model1 = model.Model()
model1.load_state_dict(weights)

# 入力の量子化パラメータ
s0 = 1 / 255
z0 = 0

# fc1 の重みの量子化パラメータ
w1 = weights["fc1.weight"].clone()
wmax = w1.max()
wmin = w1.min()
s1 = max(wmax, -wmin) * 2 / 255
z1 = 0
quantized_w1 = (w1 / s1).round()

# fc1 + relu の活性値の量子化パラメータ
# reluの出力を見て、スケールパラメータを決める
amax = 0
amin = 0
for (x, t) in train_dataloader:
	x = x.reshape(-1, 784)
	x = model1.fc1(x)
	x = torch.relu(x)
	if amax < x.max():
		amax = x.max()
sa = amax / 255

# スケールパラメータsaが決まったら、活性値の出力値を
# sa / s0 / s1 で割るようにする。そして、これをシフト演算で置き換える
M = sa / s1 / s0
shiftM = 0
while M >= 2:
	M /= 2
	shiftM += 1

# fc2 の重みの量子化パラメータ
w2 = weights["fc2.weight"].clone()
wmax = w2.max()
wmin = w2.min()
s2 = max(wmax, -wmin) * 2 / 255
z2 = 0
quantized_w2 = (w2 / s2).round()

# パラメーラの表示
print("s1 :", s1.item())
print("shiftM :", shiftM)
print("s2 :", s2.item())

# 量子化モデルの作成
qmodel = model.QuantizedModel(shiftM)
qmodel.fc1.weight.data = quantized_w1
qmodel.fc2.weight.data = quantized_w2

# 量子化モデルのテスト
total = 0
correct = 0
for (x, t) in test_dataloader:
	x = (x / s0).round()
	output = qmodel(x)
	_, predicted = torch.max(output, 1)
	correct += (predicted == t).sum()
	total += output.shape[0]
print("correct rate :", (correct / total).item())

'''
# 各層の出力の確認
for (x, t) in test_dataloader:
	x = (x / s0).round()
	x = x.reshape(-1, 784)
	x = qmodel.fc1(x)
	print(x[0])
	x = torch.relu(x)
	print(x[0])
	x = (x.int() >> shiftM).float()
	print(x[0])
	x = qmodel.fc2(x)
	print(x[0])
	break
'''
