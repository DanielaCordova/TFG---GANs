import os
import sys

import numpy as np
from PIL import Image
import os
import PIL
import glob

import torch
from torch import Tensor
from torch.nn.functional import interpolate

import Training
from StyleGAN import StyleDiscriminator
from StyleGAN.StyleGenerator import Generator
from StyleGAN.trainingStyle import load

curentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(curentdir)
sys.path.append(parentdir)
initialDepth=2
resolution = 128
out_depth2 = int(np.log2(resolution)) - 2

def adjustImgRange(data, drange_in=(-1, 1), drange_out=(0, 1)):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return torch.clamp(data, min=0, max=1)


def styleMixing(img, gen, out_depth, src_seeds, dst_seeds, style_inLocation):
    n_col = len(src_seeds)
    n_row = len(dst_seeds)
    anchura = altura = 2 ** (out_depth + 2) ##Filas y columnas
    with torch.no_grad():
        latent_size = gen.g_mapping.latent_size
        latentsIN_src = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(latent_size, ) for seed in src_seeds]).astype(np.float32))
        latentsIN_dst = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(latent_size, ) for seed in dst_seeds]).astype(np.float32))
        src_vectorW = gen.g_mapping(latentsIN_src.to("cuda"))  # [seed, layer, component] #STYLE
        dst_vectorW = gen.g_mapping(latentsIN_dst.to("cuda"))  # [seed, layer, component]
        src_images = gen.g_synthesis(src_vectorW.to("cuda"), depth=out_depth, alpha=1) ##Introducimos estilo
        dst_images = gen.g_synthesis(dst_vectorW.to("cuda"), depth=out_depth, alpha=1)

        src_vectorW_numpy = src_vectorW.cpu().numpy()
        dst_vectorW_numpy = dst_vectorW.cpu().numpy()
        canvas = Image.new('RGB', (anchura * (n_col + 1), altura * (n_row + 1)), 'white')

        ##Fruits Source B
        for col, src_image in enumerate(list(src_images)):
            src_image = adjustImgRange(src_image)
            canvas.paste(Image.fromarray(src_image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy(), 'RGB'), ((col + 1) * anchura, 0))

        ##Fruits Source A
        for row, dst_image in enumerate(list(dst_images)):
            ##Add img for the source A column
            dst_image = adjustImgRange(dst_image)
            canvas.paste(Image.fromarray(dst_image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy(), 'RGB'), (0, (row + 1) * altura))

            row_vectorW = np.stack([dst_vectorW_numpy[row]] * n_col)
            row_vectorW[:, style_inLocation[row]] = src_vectorW_numpy[:, style_inLocation[row]] ##Styles for row from Source B
            row_vectorW = torch.from_numpy(row_vectorW)

            row_images = gen.g_synthesis(row_vectorW.to("cuda"), depth=out_depth, alpha=1)  ##Img for 1 row

            for col, image in enumerate(list(row_images)): ##All img from the row with the style for each column
                image = adjustImgRange(image)
                image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                canvas.paste(Image.fromarray(image, 'RGB'), ((col + 1) * anchura, (row + 1) * altura))

        canvas = canvas.resize((400,400))
        canvas.save(img)


def main():

    nameFILE ='figure07-style-mixing.png'
    device = "cuda"
    print("Start ...")
    loadingPrev = False
    generator_FILE = "GAN_GEN_2_1.pth"
    discriminator_FILE = "GAN_DIS_2_1.pth"
    generatorOptim_FILE = "GAN_GEN_OPTIM_2_1.pth"
    discriminatorOptim_FILE = "GAN_DIS_OPTIM_2_1.pth"
    genShadow = "GAN_GEN_SHADOW_2_1.pth"
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
    generator_file = curentdir + "\\Models\\" + "\\models\\" + discriminator_FILE
    print("Loading the generator:", generator_file)

    ##gen.load_state_dict(torch.load(generator_file))

    # path for saving the files:
    # generate the images:
    # src_seeds = [639, 701, 687, 615, 1999], dst_seeds = [888, 888, 888],
    styleMixing(os.path.join(nameFILE), trainer.gen.to("cuda"),
                out_depth=initialDepth, src_seeds=[639, 1995, 6875, 6155, 1999], dst_seeds=[885, 885, 885],
                style_inLocation=[range(0, 2)] * 1 + [range(2, 7)] * 1 + [range(7, 11)] * 1)
    print('End.')


if __name__ == '__main__':
    main()
