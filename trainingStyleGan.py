
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

disc = Discriminators.DiscriminadorPorBloques(64, 'cuda')
gen = Generators.StyleNoCondGenerator(64,Constants.BATCH_SIZE,0,'cuda')

disc_opt = torch.optim.Adam(disc.parameters(), lr=Constants.LR)
gen_opt = torch.optim.Adam(gen.parameters(), lr=Constants.LR)

gen.train()
disc.train()

if not load :
    disc.apply(GANUtils.weights_init)
    gen.apply(GANUtils.weights_init)

# Carpeta para resultados

training_dir = 'porfa1Epoch'

# Dataset

ds = torch.load('preprocessedDataset.pt')

dataLoader = DataLoader(ds, batch_size=Constants.BATCH_SIZE, shuffle=True)

os.mkdir(training_dir)

# Entrenamiento

criterion = nn.BCEWithLogitsLoss()
n_epochs = 1
display_step = 2500
increase_alfa_step = 16
alfa_step = 0.0625
checkpoint_step = 1

trainer = Training.Style_Trainer(dataLoader, gen, disc, criterion, training_dir, display_step, True, True, alfa_step, increase_alfa_step, 'cuda', True, checkpoint_step
,load, load_folder, gen_load, disc_load)

trainer.train_for_epochs(n_epochs)
