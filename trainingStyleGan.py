
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

training_dir = 'BS_dif_prueba4'

# Dataset

ds = torch.load('preprocessedDataset.pt')
batch_sizes = [32, 32, 32, 16, 8, 4, 2, 1, 1]
dataLoader = DataLoader(ds, 32, shuffle=True)

os.mkdir(training_dir)

# Entrenamiento

## 67692 longitud del dataset = len
## 4231 batches de 64 imagenes = (len / batch_size) +  1 si (len % batch_size==0) = num_batches
## Deberia subir (1/4231) el alfa por cada batch = 1 / num_batches
## Va a subir (1/4231) * 25 el alfa por cada 25 batches = 1 / num_batches * increase_step

criterion = Training.HingeRelativisticLoss()
n_epochs = [4, 4, 4, 4, 8, 16, 32, 64, 64]
display_step = int(67692/Constants.BATCH_SIZE) 
increase_alfa_step = 8
alfa_step = (1/(4*len(dataLoader))) * increase_alfa_step
checkpoint_step = int(67692/Constants.BATCH_SIZE)

trainer = Training.Style_Prog_Trainer(ds, gen, disc, criterion, training_dir, True, True,'cuda', True,
load, load_folder, gen_load, disc_load, time_steps = False, time_epochs = False)

trainer.train_for_epochs(n_epochs, batch_sizes)
