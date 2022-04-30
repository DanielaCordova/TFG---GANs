import os
from torchvision.datasets import ImageFolder
import ImageFunctions
from torch.utils.data import DataLoader 
import torch

root_path = 'fruits-360_dataset/fruits-360/Training'

size = 64

ds = ImageFolder(root_path, ImageFunctions.getTransform(size))

ds_2 = [img for img,tag in ds]

torch.save(ds_2, 'preprocessedDataset.pt')