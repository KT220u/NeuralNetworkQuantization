import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim
import model
import sys

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


# BatchNormFolding (model1 -> fmodel)
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

#　パラメータの保存
torch.save(fmodel.state_dict(), "folded_weight.pth")

### fmodel を量子化

# 各relu の活性値の情報

relu1_max = 0
relu1_min = 0
relu2_max = 0
relu2_min = 0
relu3_max = 0
relu3_min = 0

for (x, t) in train_dataloader:
	x = fmodel.conv1(x)
	x = torch.relu(x)
	if relu1_max < x.max():
		relu1_max = x.max()
	x = fmodel.maxpool1(x)
	x = fmodel.conv2(x)
	x = torch.relu(x)
	if relu2_max < x.max():
		relu2_max = x.max()
	x = fmodel.maxpool2(x)
	x = x.reshape(-1, 32*8*8)
	x = fmodel.fc1(x)
	x = torch.relu(x)
	if relu3_max < x.max():
		relu3_max = x.max()

# 入力の量子化パラメータ
# 入力は -1 ~ 1 の値であるため、スケールパラメータは 1/128 とする
s_input = 1 / 128
z_input = 0

# conv1の重みの量子化パラメータ
weights = fmodel.conv1.weight.data
bias = fmodel.conv1.bias.data
z_conv1 = 0

s_conv1 = max(weights.max(), -1*weights.min()) / 127
quantized_w_conv1 = (weights / s_conv1).round()
quantized_b_conv1 = (bias / s_conv1 / s_input).round()

# conv1 + relu の活性値の量子化パラメータ
s_relu1 =  relu1_max / 255
M = s_relu1 / s_input / s_conv1
shiftM1 = 0
while M > 1:
	M /= 2
	shiftM1 += 1

# conv2の重みの量子化パラメータ
weights = fmodel.conv2.weight.data
bias = fmodel.conv2.bias.data
z_conv2 = 0

s_conv2 = max(weights .max(), weights .min() * -1) / 127
quantized_w_conv2 = (weights / s_conv2).round()
quantized_b_conv2 = (bias / s_conv2 / s_relu1).round()

# conv2 + relu の活性値の量子化パラメータ
s_relu2 = relu2_max / 255
M = s_relu2 / s_relu1 / s_conv2
shiftM2 = 0
while M > 1:
	M /= 2
	shiftM2 += 1

# fc1 の重みの量子化パラメータ
weights = fmodel.fc1.weight.data
bias = fmodel.fc1.bias.data
s_fc1 = max(weights.max(), weights.min() * -1) / 127
z_fc1 = 0
quantized_w_fc1 = (weights / s_fc1).round()
quantized_b_fc1 = (bias / s_fc1 / s_relu2).round()

# fc1 + relu の活性値の量子化パラメータ
s_relu3 =  relu3_max / 255
M = s_relu3 / s_relu2 / s_fc1
shiftM3 = 0
while M > 1:
	M /= 2
	shiftM3 += 1

# fc2 の重みの量子化パラメータ
weights = fmodel.fc2.weight.data
bias = fmodel.fc2.bias.data
s_fc2 = max(weights.max() , weights.min() * -1) / 127
z_fc2 = 0
quantized_w_fc2 = (weights / s_fc2).round()
quantized_b_fc2 = (bias / s_fc2 / s_relu3).round()

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
qmodel.conv1.bias.data = quantized_b_conv1
qmodel.conv2.weight.data = quantized_w_conv2
qmodel.conv2.bias.data = quantized_b_conv2
qmodel.fc1.weight.data = quantized_w_fc1
qmodel.fc2.weight.data = quantized_w_fc2
qmodel.fc1.bias.data = quantized_b_fc1
qmodel.fc2.bias.data = quantized_b_fc2

# テスト
correct = 0
total = 0
for (x, t) in test_dataloader:
	qmodel.eval()
	x = (x * 255).round()
	output = qmodel(x)
	_, predicted = torch.max(output, 1)
	total += output.shape[0]
	correct += (predicted == t).sum()

print("correct rate : ", (correct / total).item())

# 出力結果の確認
for (x, t) in test_dataloader:
	x = (x * 255).round()
	x = qmodel.conv1(x)
	x = torch.relu(x)
	x = (x.int() >> qmodel.shiftM1).float()
	x = qmodel.maxpool1(x)
	x = qmodel.conv2(x)
	x = torch.relu(x)
	x = (x.int() >> qmodel.shiftM2).float()
	x = qmodel.maxpool2(x)
	x = x.reshape(-1, 32*8*8)
	x = qmodel.fc1(x)
	x = torch.relu(x)
	x = (x.int() >> qmodel.shiftM3).float()
	x = qmodel.fc2(x)
	break
