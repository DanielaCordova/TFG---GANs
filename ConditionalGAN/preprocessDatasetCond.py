import sys, os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import ImageFunctions
import torch

root_path = '../fruits-360_dataset/fruits-360/Training'

classes = ['/Clementine', '/Avocado', '/Banana', '/Cocos', '/Eggplant' , '/Kaki' , '/Kiwi', '/Lemon', '/Orange', '/Pear', '/Limes', '/Raspberry', '/Watermelon', '/Mango', '/Maracuja']
size = 55

dataSets = [ImageFunctions.DataSetCarpeta(root_path + e, ImageFunctions.getTransform(size), t) for t,e in enumerate(classes)]

ds = torch.utils.data.ConcatDataset(dataSets)

torch.save(ds, 'PreprocessDatasets/preprocessed15ClassesDataset.pt')