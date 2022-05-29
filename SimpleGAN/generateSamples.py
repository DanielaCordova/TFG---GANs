import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import Training
import Constants
import torch
import torch.nn as nn
import GANUtils
import SimpleDiscriminator 
import SimpleGenerator
import sys
import os
from torch.utils.data import DataLoader




load = True
load_folder = 'GANTraining'
gen_load = 'gen_114.tar'
disc_load = 'disc_114.tar'


# Modulos
disc = SimpleDiscriminator.DiscriminadorGAN('cuda',1,40)
gen = SimpleGenerator.GeneradorGAN(40, numChan=1)


# Carpeta para resultados

training_dir = 'GANTraining'
#ños.mkdir(training_dir)

# Dataset


ds = torch.load('preprocessedMNIST8.pt')

dataLoader = DataLoader(ds, batch_size=Constants.BATCH_SIZE, shuffle=True)

criterion = nn.BCEWithLogitsLoss()
display_step = int(5000/Constants.BATCH_SIZE)
checkpoint_step = int(5000/Constants.BATCH_SIZE)
trainer = Training.Normal_Trainer(dataLoader, gen, disc, criterion,display_step, training_dir, 'cuda', True, True, checkpoint_step, load, load_folder, gen_load, disc_load, time_steps = True, time_epochs = True)
trainer.generate_samples(20)
