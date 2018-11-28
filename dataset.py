from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class BuildingGeneralizationDataset(Dataset):
    """building with/without generalization dataset."""

    def __init__(self, root_dir, transform=transforms.functional.to_tensor):
        """
         Args:
           root_dir: chemin vers le dossier avec les images
           transform: optionel fonction à appliquer pour transformer le sample
        """
        self.root = root_dir
        self.transform = transform
        self.inputName = "building_{}"
        self.targetName = self.inputName+"_output.png"
        self.inputName+=".png"
    def __len__(self):
        return int(len(os.listdir(self.root))/2) # on a n/2 paire d'images.
    def __getitem__ (self, index):
        path = self.root
        entrePath = os.path.join(path,self.inputName.format(index))
        sortiePath = os.path.join(path,self.inputName.format(index))
        sample = {"entre":Image.open(entrePath) , "sortie": Image.open(sortiePath)}
        if self.transform:
            sample = {"entre":self.transform(sample["entre"]),"sortie":self.transform(sample["sortie"])}

        return sample
        


if __name__ == '__main__':
    dataset = BuildingGeneralizationDataset("datasets\\alpe_huez_bati_50k_128")
    print(len(dataset))
    Image.fromarray(dataset[0]["entre"].numpy()).show()
    print(dataset[4]["sortie"].size)
