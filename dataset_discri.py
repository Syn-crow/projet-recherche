from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable


from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class BuildingGeneralizationDataset(Dataset):
    """building with/without generalization dataset."""

    def __init__(self, root_dir,limit_size=None, transform=[transforms.functional.to_grayscale,transforms.functional.to_tensor]):
        """
         Args:
           root_dir: chemin vers le dossier avec les images
           transform: optionel fonction à appliquer pour transformer le sample
        """
        self.root = root_dir
        self.size = limit_size
        self.transform = transform
    def __len__(self):
        if self.size == None:
            return len(os.listdir(self.root))
        else:
            return self.size
        # Pour discriminer chaque image est éligible individuelement.
    def __getitem__ (self, index):
        #on doit etre capable d'ouvrir n'importe quelle image sans discrimination
        path = self.root
        nom_fichier = os.listdir(path)[index]
        if nom_fichier.find("output") == -1:
            label = 1 # si on a l'image qui correspond bien à une vrai generalisation.
        else:
            label = 0 # si c'est un faux
        sample = {"image": Image.open(os.path.join(path, nom_fichier)) , "label": label}
        if self.transform:
            for i in range(len(self.transform)):
                sample = {"image" : self.transform[i](sample["image"]), "label" : sample['label']}

        return sample
        
print("dataset builder charge")

if __name__ == '__main__':
    dataset = BuildingGeneralizationDataset("datasets\\alpe_huez_bati_50k_128")
    print(len(dataset))
    batch_size = 10
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True);
    for j in range(3):
        for (i, item) in enumerate(train_loader):
            images = item['image']
            labels = item['label']
            try:
                images = Variable(images.float())
                print(str(i)+" epoch "+ str(j) + " OK!")
            except Exception as e:
                print("ERREUR!!! image "+str(i)+" epoch "+ str(j))
                print(image)
                print(e)
        
