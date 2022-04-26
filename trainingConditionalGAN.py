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

disc = Discriminators.DiscriminadorCondicional(3,64)
gen = Generators.GeneradorCondicional(32,131)

gen.train()
disc.train()

if not load :
    disc.apply(GANUtils.weights_init)
    gen.apply(GANUtils.weights_init)

# Carpeta para resultados

training_dir = 'ConditionalTraining'

# Dataset

ds = torch.load('preprocessedCondDataset.pt')

dataLoader = DataLoader(ds, batch_size=Constants.BATCH_SIZE, shuffle=True)

os.mkdir(training_dir)

# Entrenamiento

## 67692 longitud del dataset = len
## 4231 batches de 64 imagenes = (len / batch_size) +  1 si (len % batch_size==0) = num_batches
## Deberia subir (1/4231) el alfa por cada batch = 1 / num_batches
## Va a subir (1/4231) * 25 el alfa por cada 25 batches = 1 / num_batches * increase_step

criterion = nn.BCEWithLogitsLoss()
n_epochs = 300
display_step = int(67692/Constants.BATCH_SIZE) 
checkpoint_step = 10000

trainer = Training.Cond_Trainer(dataLoader, gen, disc, criterion,display_step, training_dir, 131, 'cuda', True, True, checkpoint_step)
trainer.train_for_epochs(n_epochs)
