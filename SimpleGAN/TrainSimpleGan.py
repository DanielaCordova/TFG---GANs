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


load_folder = None
gen_load = None
disc_load = None
load = False

if len(sys.argv) != 1 and len(sys.argv) != 5:
    print("Usage: python trainingStyleGan.py trainingDir dataset [<load> <folder> <generator_load> <discriminator_load>]")

if len(sys.argv) == 7 and sys.argv[3] == 'load' :
    load = True
    load_folder = sys.argv[4]
    gen_load = sys.argv[5]
    disc_load = sys.argv[6]
else:
    load = False

# Modulos
disc = SimpleDiscriminator.DiscriminadorGAN('cuda',1,28)
gen = SimpleGenerator.GeneradorGAN(28, numChan=1)

gen.train()
disc.train()

if not load :
    disc.apply(GANUtils.weights_init)
    gen.apply(GANUtils.weights_init)

# Carpeta para resultados

training_dir = sys.argv[1]
os.mkdir(training_dir)

# Dataset

ds = torch.load(sys.argv[2])

dataLoader = DataLoader(ds, batch_size=Constants.BATCH_SIZE, shuffle=True)

criterion = nn.BCEWithLogitsLoss()
n_epochs = 100
display_step = int(5000/Constants.BATCH_SIZE)
checkpoint_step = int(5000/Constants.BATCH_SIZE)
trainer = Training.Normal_Trainer(dataLoader, gen, disc, criterion,display_step, training_dir, 'cuda', True, True, checkpoint_step, time_steps = True, time_epochs = True)
trainer.train_for_epochs(n_epochs)
