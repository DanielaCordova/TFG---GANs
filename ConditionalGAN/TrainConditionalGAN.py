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

if len(sys.argv) != 2 and len(sys.argv) != 6:
    print("Usage: python trainingStyleGan.py trainingDir <load> <folder> <generator_load> <discriminator_load>")

if len(sys.argv) == 5 and sys.argv[2] == 'load' :
    load = True
    load_folder = sys.argv[3]
    gen_load = sys.argv[4]
    disc_load = sys.argv[5]
else:
    load = False

# Modulos
disc = Discriminators.DiscriminadorCondicional('cuda',18,64)
gen = Generators.GeneradorCondicional(64,15)

gen.train()
disc.train()

if not load :
    disc.apply(GANUtils.weights_init)
    gen.apply(GANUtils.weights_init)

# Carpeta para resultados

training_dir = sys.argv[2]
os.mkdir(training_dir)

# Dataset

ds = torch.load('preprocessed15ClassesCondDataset.pt')

dataLoader = DataLoader(ds, batch_size=Constants.BATCH_SIZE, shuffle=True)

os.mkdir(training_dir)

# Entrenamiento

criterion = nn.BCEWithLogitsLoss()
n_epochs = 100
display_step = int(7230/Constants.BATCH_SIZE) 
checkpoint_step = int(7230/Constants.BATCH_SIZE) 

trainer = Training.Cond_Trainer(dataLoader, gen, disc, criterion,display_step, training_dir, 15, 'cuda', True, True, checkpoint_step)
trainer.train_for_epochs(n_epochs)
