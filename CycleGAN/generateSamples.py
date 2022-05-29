import sys, os

curentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(curentdir)
sys.path.append(parentdir)

import CycleGenerator
import CycleDiscriminator
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
gen_AB = CycleGenerator.Generator(3, 3).to(device)
gen_BA = CycleGenerator.Generator(3, 3).to(device)
gen_opt = torch.optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=lr, betas=(0.5, 0.999))
disc_A = CycleDiscriminator.Discriminator(3).to(device)
disc_A_opt = torch.optim.Adam(disc_A.parameters(), lr=lr, betas=(0.5, 0.999))
disc_B = CycleDiscriminator.Discriminator(3).to(device)
disc_B_opt = torch.optim.Adam(disc_B.parameters(), lr=lr, betas=(0.5, 0.999))


adv_criterion = nn.MSELoss()
recon_criterion = nn.L1Loss()

# Carpeta para resultados

training_dir = 'CycleTraining'
load_dir = 'models'
gen_disc_load = 'cycleGAN.pth'

# Dataset

ds1 = torch.load('PreprocessDatasets/preprocessedApple.pt')
ds2 = torch.load('PreprocessDatasets/preprocessedPear.pt')

dataLoader1 = DataLoader(ds1, batch_size=Constants.BATCH_SIZE, shuffle=True)
dataLoader2 = DataLoader(ds2, batch_size=Constants.BATCH_SIZE, shuffle=True)

criterion = nn.BCEWithLogitsLoss()
display_step = 1000
checkpoint_step = 1000


trainer = Training.Cycle_Trainer(dataLoader1, dataLoader2, gen_AB, gen_BA, gen_opt, disc_A, disc_A_opt, disc_B, disc_B_opt, adv_criterion,
recon_criterion, display_step, training_dir, target_shape, 'cuda', True, checkpoint_step, True, load_dir, gen_disc_load, time_steps = True, time_epochs = True)

trainer.generate_samples()