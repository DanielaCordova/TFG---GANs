import sys, os

import torch
from torch.nn.functional import interpolate

import Training
from StyleGAN import StyleDiscriminator
from StyleGAN.Components import make_dataset, make_logger



import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torchvision.utils import save_image

from StyleGAN.StyleGenerator import Generator
from StyleGAN.generateStyleMixing import adjustImgRange
from StyleGAN.trainingStyle import load

curentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(curentdir)
sys.path.append(parentdir)

output_dir = curentdir + "\\Models\\"
img_dir = "C:/Users/Daniela/Documents/TFG/fruits-360_dataset/fruits-360\Training"
logger = make_logger("project", output_dir, 'log')


if __name__ == '__main__':

    loadingPrev = False
    generator_FILE = "GAN_GEN_2_4.pth"
    discriminator_FILE = "GAN_DIS_2_4.pth"
    generatorOptim_FILE = "GAN_GEN_OPTIM_2_4.pth"
    discriminatorOptim_FILE = "GAN_DIS_OPTIM_2_4.pth"
    genShadow = "GAN_GEN_SHADOW_2_4.pth"
    initialDepth =2

    # gen = Generator(
    #     resolution=120,
    #     conditional=False,
    #     n_classes=131).to('cuda')

    trainer = Training.Style_Prog_Trainer(
        generator=Generator,
        discriminator=StyleDiscriminator.Discriminator,
        conditional=False,
        n_classes=131,
        resolution=128,
        num_channels=3,
        latent_size=512,
        loss="logistic",
        drift=0.001,
        d_repeats=1,
        use_ema=True,
        ema_decay=0.999,
        device='cuda',
        checksave=False,
        load=False,
        load_dir=None,
        gen_load=None,
        disc_load=None,
        time_steps=True,
        time_epochs=True)

    load(trainer.gen, generator_FILE)
    trainer.dis.load_state_dict(torch.load(curentdir + "\\Models\\" + "\\models\\" + discriminator_FILE))
    load(trainer.gen_shadow, genShadow)
    trainer.gen_optim.load_state_dict(torch.load(curentdir + "\\Models\\" + "\\models\\" + generatorOptim_FILE))
    trainer.dis_optim.load_state_dict(torch.load(curentdir + "\\Models\\" + "\\models\\" + discriminatorOptim_FILE))
    ##gen.load_state_dict()

    # path for saving the files:
    num_samples = 5
    os.makedirs(output_dir, exist_ok=True)
    latent_size = 512
    resolution = 128
    out_depth = int(np.log2(resolution)) - 2

    for img_num in tqdm(range(1, num_samples + 1)):
        # generate the images:
        with torch.no_grad():
            point = torch.randn(1, latent_size)
            point = (point / point.norm()) * (pow(latent_size, 0.5))

            ##Generamos imagen
            fakeImage = trainer.gen(point.to("cuda"), depth=initialDepth, alpha=1).to("cuda")
            # Ajustamos Colores
            fakeImage = adjustImgRange(fakeImage)

        # Guardar imagen
        scale_factor = scale_factor=int(np.power(2, out_depth - initialDepth - 1))
        if(scale_factor >1):
            fakeImage= interpolate(fakeImage, scale_factor=scale_factor)
        save_image(fakeImage, os.path.join(output_dir, str(img_num) + ".png"))
        ##save_image(fakeImage,os.path.join(output_dir, str(img_num) + ".png") )

    print("Generated %d images at %s" % (num_samples, output_dir))
