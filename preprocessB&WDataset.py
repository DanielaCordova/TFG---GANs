import os
from torchvision.datasets import ImageFolder
import ImageFunctions
from torch.utils.data import DataLoader 
import torch

root_path = 'D:/UNI/Mnist/MNIST - JPG - training'

classes = ['/8']
size = 64

dataSets = [ImageFunctions.DataSetCarpetaBlackandWhite(root_path + e, ImageFunctions.getTransform(size), t) for t,e in enumerate(classes)]

ds = torch.utils.data.ConcatDataset(dataSets)

torch.save(ds, 'preprocessedMNIST8.pt')