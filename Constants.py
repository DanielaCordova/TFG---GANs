
import os
import torch

Z_DIM = 512
NUM_CHANNELS = 3
TRUNCATION = 0.7
EJEMPLOSTEST = 512
LR = 0.0002
DATA_DIR = os.path.sep + 'Training'
DEVICE = torch.device('cuda')
DIMIMG = (3,64,64)
N_EPOCHS = 200
DISPLAY_STEP = 400
BATCH_SIZE = 64
INITIAL_ALPHA = 0.1


