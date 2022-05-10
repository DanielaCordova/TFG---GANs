import GANUtils
import Generators
import Discriminators
import Training
import torch
import torch.nn as nn
import Constants
from torchvision.datasets import ImageFolder
import ImageFunctions
from torch.utils.data import DataLoader
import os
import sys


load_folder = None
gen_load = None
disc_load = None
load = False

if len(sys.argv) != 1 and len(sys.argv) != 5:
    print("Usage: python trainingStyleGan.py <load> <folder> <generator_load> <discriminator_load>")

if len(sys.argv) == 5 and sys.argv[1] == 'load' :
    load = True
    load_folder = sys.argv[2]
    gen_load = sys.argv[3]
    disc_load = sys.argv[4]
else:
    load = False

# Modulos
disc = Discriminators.DiscriminadorCondicional('cuda',3,64)
gen = Generators.GeneradorGAN(64)

gen.train()
disc.train()

if not load :
    disc.apply(GANUtils.weights_init)
    gen.apply(GANUtils.weights_init)

# Carpeta para resultados

training_dir = 'GANTraining17'

# Dataset

ds = torch.load('preprocessedMNIST.pt')

dataLoader = DataLoader(ds, batch_size=Constants.BATCH_SIZE, shuffle=True)

criterion = nn.BCEWithLogitsLoss()
n_epochs = 100
display_step = int(2900/Constants.BATCH_SIZE)
checkpoint_step = int(2900/Constants.BATCH_SIZE)

trainer = Training.Normal_Trainer(dataLoader, gen, disc, criterion,display_step, training_dir, 'cuda', True, True, checkpoint_step, time_steps = True, time_epochs = True)
trainer.train_for_epochs(n_epochs)
