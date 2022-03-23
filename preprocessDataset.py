import os
from torchvision.datasets import ImageFolder
import ImageFunctions
from torch.utils.data import DataLoader 
import torch

ds_aux = ImageFolder('fruits-360_dataset/fruits-360/Training', transform=ImageFunctions.getTransform())
ds = [img for img, tag in ds_aux]

torch.save(ds, 'preprocessedDataset.pt')

exit()