import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import model
import sys

test_dataset = torchvision.datasets.MNIST(root='../data', train=False, transform=transforms.ToTensor(), download = True)
batch_size = 64
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

model = model.Model()
model.load_state_dict(torch.load("weight.pth"))


correct = 0
total = 0
for (x, t) in test_dataloader:
	output = model(x)
	_, predicted = torch.max(output, 1)
	total += output.shape[0]
	correct += (predicted == t).sum()
print("correct rate : ", (correct / total).item())
