import torch
import model

model1 = model.Model()
model1.load_state_dict(torch.load("weight.pth"))

print(model1.state_dict())
# 入力の量子化パラメータ
s0 = 1 / 255
z0 = 0

# fc1 の重みの量子化パラメータ

# fc1 + relu の活性値の量子化パラメータ
shiftM = 1
# fc2 の重みの量子化パラメータ

qmodel = model.QuantizedModel(shiftM)

