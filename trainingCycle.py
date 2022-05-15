import GANUtils
import Generators
import Discriminators
import torch
import torch.nn as nn
import Constants
from torchvision.datasets import ImageFolder
import ImageFunctions
from torch.utils.data import DataLoader
import os
import sys
import torch.nn.functional as F

dim_A = 3
dim_B = 3
device = 'cuda'
n_epochs = 100
display_step = 1000
batch_size =1
lr = 0.0002
load_shape = 200
target_shape = 200
device = 'cuda'

# Modulos
gen_AB = Generators.CycleGenerator(dim_A, dim_B).to(device)
gen_BA = Generators.CycleGenerator(dim_B, dim_A).to(device)
gen_opt = torch.optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=lr, betas=(0.5, 0.999))
disc_A = Discriminators.CycleDiscriminator(dim_A).to(device)
disc_A_opt = torch.optim.Adam(disc_A.parameters(), lr=lr, betas=(0.5, 0.999))
disc_B = Discriminators.CycleDiscriminator(dim_B).to(device)
disc_B_opt = torch.optim.Adam(disc_B.parameters(), lr=lr, betas=(0.5, 0.999))


adv_criterion = nn.MSELoss() 
recon_criterion = nn.L1Loss() 

# Carpeta para resultados

training_dir = 'D:/UNI/TFG/CycleTraining/CycleTraining'
load_dir = 'D:/UNI/TFG/CycleTraining/CycleTraining'
gen_disc_load = 'cycleGAN_1000.pth'

# Dataset

ds1 = torch.load('preprocessedApple.pt')
ds2 = torch.load('preprocessedBanana.pt')

dataLoader1 = DataLoader(ds1, batch_size=Constants.BATCH_SIZE, shuffle=True)
dataLoader2 = DataLoader(ds2, batch_size=Constants.BATCH_SIZE, shuffle=True)

criterion = nn.BCEWithLogitsLoss()
n_epochs = 100
display_step = int(2900/Constants.BATCH_SIZE)
checkpoint_step = int(2900/Constants.BATCH_SIZE)


trainer = Cycle_Trainer(dataLoader1, dataLoader2, gen_AB, gen_BA, gen_opt, disc_A, disc_A_opt, disc_B, disc_B_opt, adv_criterion, 
recon_criterion, display_step, training_dir, target_shape, 'cuda', True, checkpoint_step, False, load_dir, gen_disc_load, time_steps = True, time_epochs = True)


trainer.train_for_epochs(n_epochs)
