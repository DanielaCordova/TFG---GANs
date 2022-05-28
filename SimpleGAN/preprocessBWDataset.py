import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from torchvision.datasets import ImageFolder
import ImageFunctions
from torch.utils.data import DataLoader 
from torchvision import transforms
import torch


root_path = '../MNIST Dataset JPG format/MNIST - JPG - training'

classes = ['/8']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(56)
])

dataSets = [ImageFunctions.DataSetCarpetaBlackandWhite(root_path + e, transform, t) for t,e in enumerate(classes)]

ds = torch.utils.data.ConcatDataset(dataSets)

torch.save(ds, 'preprocessedMNIST8(1).pt')