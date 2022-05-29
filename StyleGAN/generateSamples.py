import sys, os

curentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(curentdir)
sys.path.append(parentdir)

import Training
import torch
import torch.nn as nn
import Constants
from torch.utils.data import DataLoader

n_epochs = 100
batch_size = 1
lr = 0.0002
target_shape = 64
device = 'cuda'

# Modulos

adv_criterion = nn.MSELoss()
recon_criterion = nn.L1Loss()

# Carpeta para resultados

training_dir = 'CycleTraining'
load_dir = 'CycleTraining'
gen_disc_load = 'cycleGAN_0.pth'

# Dataset

ds1 = torch.load('PreprocessDatasets/preprocessedApple.pt')
ds2 = torch.load('PreprocessDatasets/preprocessedBanana.pt')

dataLoader1 = DataLoader(ds1, batch_size=Constants.BATCH_SIZE, shuffle=True)
dataLoader2 = DataLoader(ds2, batch_size=Constants.BATCH_SIZE, shuffle=True)

criterion = nn.BCEWithLogitsLoss()
display_step = 1000
checkpoint_step = 1000

