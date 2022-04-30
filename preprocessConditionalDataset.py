import os
from torchvision.datasets import ImageFolder
import ImageFunctions
from torch.utils.data import DataLoader 
import torch

root_path = 'fruits-360_dataset/fruits-360/Training'

classes = ['/Clementine', '/Avocado', '/Banana', '/Cocos', '/Eggplant' , '/Kaki' , '/Kiwi', '/Lemon', '/Orange', '/Pear', '/Limes', '/Raspberry', '/Watermelon', '/Mango', '/Maracuja']
size = 55

dataSets = [ImageFunctions.DataSetCarpeta(root_path + e, ImageFunctions.getTransform(size), t) for t,e in enumerate(classes)]

ds = torch.utils.data.ConcatDataset(dataSets)

torch.save(ds, 'preprocessed15ClassesCondDataset.pt')