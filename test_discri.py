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

import datetime

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
if __name__ == '__main__':
    num_epochs = 5;
    batch_size = 100;
    learning_rate = 0.001;

    labels_map = {0 : "pas generalise", 1 : "generalise"}
    #chargement du dataset

    dataset = BuildingGeneralizationDataset("datasets\\alpe_huez_bati_50k_128")
    dataset2 = BuildingGeneralizationDataset("datasets\\alpe_huez_bati_50k_128")

    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True);
    test_loader = torch.utils.data.DataLoader(dataset=dataset2,
                                              batch_size=batch_size,
                                              shuffle=True);
    #instance of the Conv Net
    cnn = CNN();
    #loss function and optimizer
    criterion = nn.CrossEntropyLoss();
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate);
    losses = [];
    for epoch in range(num_epochs):
        print("eh poc")
        
        for (i, item) in enumerate(train_loader):
            images = item['image']
            labels = item['label']
            try:
                images = Variable(images.float())
            except Exception as e:
                print(images)
                print(e)
            labels = Variable(labels)
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.data[0]);
            if (i+1) % 3 == 0:
                print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' 
                       %(epoch+1, num_epochs, i+1, len(dataset)//batch_size, loss.data[0]))
    
    cnn.eval()
    correct = 0
    total = 0
    now = datetime.datetime.now()
    torch.save(cnn.state_dict(), "{}_{}_{}__{}_{}".format(now.day,now.month,now.year,now.hour,now.minute)+".pt")
    print("saved")
    for(i, item) in enumerate(train_loader):
        images = item['image']
        labels = item['label']
        images = Variable(images.float())
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Test Accuracy of the model on the 10000 test images: %.4f %%' % (100 * correct / total))

