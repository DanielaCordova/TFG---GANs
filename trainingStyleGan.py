
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

disc = Discriminators.DiscriminadorPorBloques(64,3,512,'cuda')
gen = Generators.StyleNoCondGenerator(64,Constants.BATCH_SIZE,0,'cuda')

disc_opt = torch.optim.Adam(disc.parameters(), lr=Constants.LR)
gen_opt = torch.optim.Adam(gen.parameters(), lr=Constants.LR)

gen.train()
disc.train()

if not load :
    disc.apply(GANUtils.weights_init)
    gen.apply(GANUtils.weights_init)

# Carpeta para resultados

training_dir = 'trainingPaperLike-GenPaper-DiscNuestrov2'

# Dataset

ds = torch.load('preprocessedDataset.pt')

dataLoader = DataLoader(ds, batch_size=Constants.BATCH_SIZE, shuffle=True)

os.mkdir(training_dir)

# Entrenamiento

## 67692 longitud del dataset = len
## 2116 batches de 64 imagenes = (len / batch_size) +  1 si (len % batch_size==0) = num_batches
## Deberia subir (1/2116) el alfa por cada batch = 1 / num_batches
## Va a subir (1/2116) * 25 el alfa por cada 25 batches = 1 / num_batches * increase_step

criterion = nn.BCEWithLogitsLoss()
n_epochs = [5,5,5,5,64]
display_step = 5000
increase_alfa_step = 25
alfa_step = (1/len(dataLoader)) * increase_alfa_step
checkpoint_step = 10000

trainer = Training.Style_Prog_Trainer(dataLoader, gen, disc, criterion, training_dir, display_step, True, True, increase_alfa_step, alfa_step, 'cuda', True, checkpoint_step
,load, load_folder, gen_load, disc_load)

trainer.train_for_epochs(n_epochs)
