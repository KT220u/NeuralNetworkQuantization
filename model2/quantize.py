import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim
import model
import sys

train_dataset = datasets.MNIST(root='../data', train=True, transform=transforms.ToTensor(), download = True)
test_dataset = datasets.MNIST(root='../data', train=False, transform=transforms.ToTensor(), download = True)
batch_size = 64
train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

weights = torch.load("weight.pth")
model1 = model.Model()
model1.load_state_dict(weights)

# 各relu の活性値の情報

relu1_max = 0
relu1_min = 0
relu2_max = 0
relu2_min = 0
relu3_max = 0
relu3_min = 0

for (x, t) in train_dataloader:
	x = model1.conv1(x)
	x = torch.relu(x)
	if relu1_max < x.max():
		relu1_max = x.max()
	x = model1.maxpool1(x)
	x = model1.conv2(x)
	x = torch.relu(x)
	if relu2_max < x.max():
		relu2_max = x.max()
	x = model1.maxpool2(x)
	x = x.reshape(-1, 32*4*4)
	x = model1.fc1(x)
	x = torch.relu(x)
	if relu3_max < x.max():
		relu3_max = x.max()

# 入力の量子化パラメータ
s_input = 1 / 255
z_input = 0

# conv1の重みの量子化パラメータ
weights = model1.conv1.weight.data
s_conv1 = max(weights.max(), -1 * weights.min()) / 127
z_conv1 = 0
quantized_w_conv1 = (weights / s_conv1).round()

# conv1 + relu の活性値の量子化パラメータ
s_relu1 =  relu1_max / 255
M = s_relu1 / s_input / s_conv1
shiftM1 = 0
while M >= 2:
	M /= 2
	shiftM1 += 1

# conv2の重みの量子化パラメータ
weights = model1.conv2.weight.data
s_conv2 = max(weights.max(), -1 * weights.min()) / 127
z_conv2 = 0
quantized_w_conv2 = (weights / s_conv2).round()

# conv2 + relu の活性値の量子化パラメータ
s_relu2 = relu2_max / 255
M = s_relu2 / s_relu1 / s_conv2
shiftM2 = 0
while M >= 2:
	M /= 2
	shiftM2 += 1

# fc1 の重みの量子化パラメータ
weights = model1.fc1.weight.data
s_fc1 = max(weights.max(), weights.min() * -1) / 127
z_fc1 = 0
quantized_w_fc1 = (weights / s_fc1).round()

# fc1 + relu の活性値の量子化パラメータ
s_relu3 =  relu3_max / 255
M = s_relu3 / s_relu2 / s_fc1
shiftM3 = 0
while M >= 2:
	M /= 2
	shiftM3 += 1

# fc2 の重みの量子化パラメータ
weights = model1.fc2.weight.data
s_fc2 = max(weights.max() , weights.min() * -1) / 127
z_fc2 = 0
quantized_w_fc2 = (weights / s_fc2).round()

# パラメーラの表示
print("s_conv1 :", s_conv1)
print("s_conv2 :", s_conv2)
print("s_fc1 :", s_fc1)
print("s_fc2 :", s_fc2)
print("shiftM1 :", shiftM1)
print("shiftM2 :", shiftM2)
print("shiftM3 :", shiftM3)

# 量子化モデルの作成
qmodel = model.QuantizedModel(shiftM1, shiftM2, shiftM3)
qmodel.conv1.weight.data = quantized_w_conv1
qmodel.conv2.weight.data = quantized_w_conv2
qmodel.fc1.weight.data = quantized_w_fc1
qmodel.fc2.weight.data = quantized_w_fc2

# テスト
correct = 0
total = 0
for (x, t) in test_dataloader:
	x = (x * 255).round()
	output = qmodel(x)
	_, predicted = torch.max(output, 1)
	total += output.shape[0]
	correct += (predicted == t).sum()
print("correct rate : ", (correct / total).item())

'''
# 出力結果の確認
x = qmodel.conv1(x)
x = torch.relu(x)
x = (x.int() >> qmodel.shiftM1).float()
x = qmodel.maxpool1(x)
x = qmodel.conv2(x)
x = torch.relu(x)
x = (x.int() >> qmodel.shiftM2).float()
x = qmodel.maxpool2(x)
x = x.reshape(-1, 32*4*4)
x = qmodel.fc1(x)
x = torch.relu(x)
x = (x.int() >> qmodel.shiftM3).float()
print(x[0])
x = qmodel.fc2(x)
'''
