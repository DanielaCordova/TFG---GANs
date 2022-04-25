import os
from torchvision.datasets import ImageFolder
import ImageFunctions
from torch.utils.data import DataLoader 
import torch

ds_aux = ImageFolder('fruits-360_dataset/fruits-360/Training', transform=ImageFunctions.getTransform(64))

torch.save(ds_aux, 'preprocessedCondDataset.pt')

exit()