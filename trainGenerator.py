
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

# Modulos

disc = Discriminators.DiscriminadorLibro((3,64,64), 0, 'cuda')
gen = Generators.EqualizedStyleGen(512, 3, (512,4,4), 8, (3,64,64), 64, 'cuda')

disc_opt = torch.optim.Adam(disc.parameters(), lr=Constants.LR)
gen_opt = torch.optim.Adam(gen.parameters(), lr=Constants.LR)

disc.apply(GANUtils.weights_init)
gen.apply(GANUtils.weights_init)

# Dataset

ds = torch.load('preprocessedDataset.pt')

dataLoader = DataLoader(ds, batch_size=Constants.BATCH_SIZE, shuffle=True)

# Carpeta para resultados

training_dir = 'training-EqualizedGen'

os.mkdir(training_dir)

# Entrenamiento

criterion = nn.BCEWithLogitsLoss()
n_epochs = 1000000
display_step = 50
increase_alfa_step = 100
alfa_step = 0.02

trainer = Training.Style_Trainer(dataLoader, gen, disc, criterion, training_dir, display_step, True, False, alfa_step, increase_alfa_step, 'cuda')
trainer.train_generator_for_epochs(n_epochs)

