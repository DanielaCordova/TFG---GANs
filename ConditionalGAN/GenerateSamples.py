
import sys, os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import ConditionalGenerator
import ConditionalDiscriminator
import Training
import torch
import torch.nn as nn
import Constants
from torch.utils.data import DataLoader 
import os
import sys


if len(sys.argv) != 5:
    print("Usage: python GenerateSamples.py resultsDir <folder> <generator_load> <discriminator_load>")


load = True
load_folder = sys.argv[2]
gen_load = sys.argv[3]
disc_load = sys.argv[4]


# Modulos
disc = ConditionalDiscriminator.DiscriminadorCondicional('cuda',18,64)
gen = ConditionalGenerator.GeneradorCondicional(64,15)

# Carpeta para resultados

training_dir = sys.argv[1]
os.mkdir(training_dir)

# Dataset

dataLoader = None


# Entrenamiento

criterion = nn.BCEWithLogitsLoss()
display_step = int(7230/Constants.BATCH_SIZE) 
checkpoint_step = int(7230/Constants.BATCH_SIZE) 

trainer = Training.Cond_Trainer(dataLoader, gen, disc, criterion,display_step, training_dir,
 15, 'cuda', True, True, checkpoint_step, load=True, load_dir = load_folder, gen_load = gen_load,
 disc_load = disc_load)
trainer.generate_samples(10)
