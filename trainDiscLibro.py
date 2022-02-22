import Constants
import Discriminators
import torch
import GANUtils
import ImageFunctions
from torch.utils.data import DataLoader 
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder
import torch.nn as nn
import Training
import os
import shutil

# Inicializar discriminador

disc = Discriminators.DiscriminadorLibro((3,64,64), 0)
disc_opt = torch.optim.Adam(disc.parameters(), lr = Constants.LR)

disc.apply(GANUtils.weights_init)

# Cargar Dataset


dsk = ImageFunctions.DataSetCarpeta('fruits-360_dataset/fruits-360/Training/Kiwi', ImageFunctions.getTransform(), 0)
dsg = ImageFunctions.DataSetCarpeta('fruits-360_dataset/fruits-360/Training/Guava', ImageFunctions.getTransform(), 1)

dsTotal = ConcatDataset([dsk, dsg])

print(dsTotal.__len__())

dl = DataLoader(dsTotal, batch_size=200, shuffle=True)

# Parametros de training

device = Constants.DEVICE
criterion = nn.BCEWithLogitsLoss()
display_step = Constants.DISPLAY_STEP
increase_alpha_step = 100000000
n_epoch = 10000


os.mkdir('training-disc-libro-3')


Training.train_discriminator(disc,disc_opt, dl, n_epoch, device, criterion, 
                                display_step, increase_alpha_step, 0, 'training-disc-libro-3')



