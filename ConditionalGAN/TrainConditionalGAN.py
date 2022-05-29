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
import ConditionalDiscriminator 
import ConditionalGenerator
import os
from torch.utils.data import DataLoader

load_folder = None
gen_load = None
disc_load = None
load = False

if len(sys.argv) != 1 and len(sys.argv) != 5:
    print("Usage: python trainingStyleGan.py trainingDir <load> <folder> <generator_load> <discriminator_load>")

if len(sys.argv) == 5 and sys.argv[2] == 'load' :
    load = True
    load_folder = sys.argv[3]
    gen_load = sys.argv[4]
    disc_load = sys.argv[5]
else:
    load = False

# Modulos
disc = ConditionalDiscriminator.DiscriminadorCondicional('cuda',18,64)
gen = ConditionalGenerator.GeneradorCondicional(64,15)

gen.train()
disc.train()

if not load :
    disc.apply(GANUtils.weights_init)
    gen.apply(GANUtils.weights_init)

# Carpeta para resultados

training_dir = sys.argv[1]
os.mkdir(training_dir)

# Dataset

ds = torch.load('PreprocessDatasets/preprocessed15ClassesDataset.pt')

dataLoader = DataLoader(ds, batch_size=Constants.BATCH_SIZE, shuffle=True)


# Entrenamiento

criterion = nn.BCEWithLogitsLoss()
n_epochs = 100
display_step = int(7230/Constants.BATCH_SIZE) 
checkpoint_step = int(7230/Constants.BATCH_SIZE) 

trainer = Training.Cond_Trainer(dataLoader, gen, disc, criterion,display_step, training_dir, 15, 'cuda', True, True, checkpoint_step)
trainer.train_for_epochs(n_epochs)
