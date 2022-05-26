import os
from torchvision.datasets import ImageFolder
import ImageFunctions
from torch.utils.data import DataLoader 
from torchvision import transforms
import torch

root_path = 'Mnist/MNIST - JPG - training'

classes = ['/8']

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataSets = [ImageFunctions.DataSetCarpetaBlackandWhite(root_path + e, transform, t) for t,e in enumerate(classes)]

ds = torch.utils.data.ConcatDataset(dataSets)

torch.save(ds, 'PreprocessDatasets/preprocessedMNIST8.pt')