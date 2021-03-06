import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np;
from torch.utils.data import Dataset, DataLoader
import random;
import math;
import torch.nn.functional as F

import dataset_discri
from dataset_discri import BuildingGeneralizationDataset

from PIL import Image

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(32768, 100)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
model = CNN()
model.load_state_dict(torch.load("21_11_18.pt"))
model.eval()

dataset2 = BuildingGeneralizationDataset("datasets\\lourdes_bati_50k_128")
batch_size = 1
test_loader = torch.utils.data.DataLoader(dataset=dataset2,
                                              batch_size=batch_size,
                                              shuffle=True);

correct = 0
total = 0

with torch.no_grad():
    print(str(correct)+"/"+str(total))

    for(i, item) in enumerate(test_loader):
        
        try:
            images = item['image']
            labels = item['label']
            outputs = model(images)
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += int(torch.eq(predicted, labels)[0])
            if(i%5==0):
                print(str(correct)+"/"+str(total))
        except Exception as e:
            print(e)

print('Accuracy of the network on the {} test images: {}'.format(total,100 * correct / total))
