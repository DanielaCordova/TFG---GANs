import copy
import datetime
import os
import random
import time
import timeit
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
#from data import get_data_loader
from torch.nn.functional import interpolate
from torch.nn.modules.sparse import Embedding

import Losses as Losses
#from models import update_average
from Blocks import (DiscriminatorBlock, DiscriminatorTop,
                           GSynthesisBlock, InputBlock, Conv2dPropia)
from CustomLayers import (EqualizedConv2d, EqualizedLinear,
                                 PixelNormLayer, Truncation, EqualizedLinearPropia)


class GMapping(nn.Module):

    def __init__(self, latent_size=512, dlatent_size=512, dlatent_broadcast=None,
                 mapping_layers=8, mapping_fmaps=512, mapping_lrmul=0.01, mapping_nonlinearity='lrelu',
                 use_wscale=True, normalize_latents=True, **kwargs):

        super().__init__()

        self.latent_size = latent_size
        self.mapping_fmaps = mapping_fmaps
        self.dlatent_size = dlatent_size
        self.dlatent_broadcast = dlatent_broadcast

        layers = []

        if normalize_latents: ##normalizamos el vector de características en cada píxel a la longitud de la unidad en el generador después de cada capa convolucional
            layers.append(PixelNormLayer())


        layers.append(EqualizedLinearPropia(self.latent_size, self.mapping_fmaps,gain=np.sqrt(2), lrmul=mapping_lrmul))
        layers.append(nn.LeakyReLU(negative_slope=0.2))

        for layer in range(1, mapping_layers): ##las otras 8 capas
            fmaps_in = self.mapping_fmaps
            fmaps_out = self.dlatent_size if layer == mapping_layers - 1 else self.mapping_fmaps
            layers.append(EqualizedLinearPropia(fmaps_in, fmaps_out, gain=np.sqrt(2), lrmul=mapping_lrmul))
            layers.append(nn.LeakyReLU(negative_slope=0.2))

        self.map = nn.Sequential(*layers)

    def forward(self, x):
        x = self.map(x)

        # Broadcast -> batch_size * dlatent_broadcast * dlatent_size VA PEOR SI LO QUITAS!!!
        if self.dlatent_broadcast is not None:
            x = x.unsqueeze(1).expand(-1, self.dlatent_broadcast, -1)
        return x


class GSynthesis(nn.Module):

    def __init__(self, dlatent_size=512, num_channels=3, resolution=1024,
                 fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 use_styles=True, const_input_layer=True, use_noise=True, nonlinearity='lrelu',
                 use_wscale=True, use_pixel_norm=False, use_instance_norm=True, blur_filter=None,
                 structure='linear', **kwargs):

        super().__init__()

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.structure = structure

        ##Calculo de la resolucion, produndiad y capas
        resolution_log2 = int(np.log2(resolution))
        self.depth = resolution_log2 - 1
        self.num_layers = resolution_log2 * 2 - 2
        self.num_styles = self.num_layers if use_styles else 1  ##Numero de estilos posibles

        self.init_block = InputBlock(nf(1), dlatent_size)
        # create the ToRGB layers for various outputs
        rgb_converters = [Conv2dPropia(nf(1), num_channels, gain=1, kernel_size=1)]

        # Building blocks for remaining layers.
        layers = []
        for res in range(3, resolution_log2 + 1):
            last_channels = nf(res - 2)
            channels = nf(res - 1)
            layers.append(GSynthesisBlock(last_channels, channels, dlatent_size))
            rgb_converters.append(Conv2dPropia(channels, num_channels, gain=1))

        self.bloques = nn.ModuleList(layers)
        self.to_rgb = nn.ModuleList(rgb_converters)  ##ThetoRGBrepresents  a  layer  that  projects  feature  vectors  to  RGB  colors  andfromRGBdoesthe reverse

        # register the temporary upsampler
        self.temporaryUpsampler = lambda x: interpolate(x, scale_factor=2)

    def forward(self, dlatents_in, depth=0, alpha=0., labels_in=None):
        x = self.init_block(dlatents_in[:, 0:2])

        if depth > 0:
            for i, bloque in enumerate(self.bloques[:depth - 1]):
                x = bloque(x, dlatents_in[:, 2 * (i + 1):2 * (i + 2)])

            residual = self.to_rgb[depth - 1](self.temporaryUpsampler(x))
            straight = self.to_rgb[depth](self.bloques[depth - 1](x, dlatents_in[:, 2 * depth:2 * (depth + 1)]))

            images_out = (alpha * straight) + ((1 - alpha) * residual)
        else:
            images_out = self.to_rgb[0](x)

        return images_out


class Generator(nn.Module):

    def __init__(self, resolution, latent_size=512, dlatent_size=512,
                 conditional=False, n_classes=0, truncation_psi=0.7,
                 truncation_cutoff=8, dlatent_avg_beta=0.995,
                 style_mixing_prob=0.9, **kwargs):

        super(Generator, self).__init__()

        if conditional:
            self.class_embedding = nn.Embedding(n_classes, latent_size)
            latent_size *= 2

        self.conditional = conditional
        self.style_mixing_prob = style_mixing_prob

        # Creamos los componentes del generador
        self.num_layers = (int(np.log2(resolution)) - 1) * 2
        self.g_mapping = GMapping(latent_size, dlatent_size, dlatent_broadcast=self.num_layers, **kwargs)
        self.g_synthesis = GSynthesis(resolution=resolution, **kwargs)

        if truncation_psi > 0:
            self.truncation = Truncation(avg_latent=torch.zeros(dlatent_size),
                                         max_layer=truncation_cutoff,
                                         threshold=truncation_psi,
                                         beta=dlatent_avg_beta)
        else:
            self.truncation = None

    def forward(self, latents_in, depth, alpha, labels_in=None): #latents_in=noise
        if not self.conditional:
            if labels_in is not None:
                warnings.warn(
                    "Generator is unconditional, labels_in will be ignored")
        else: ##Para gerador condicional
            embedding = self.class_embedding(labels_in)
            latents_in = torch.cat([latents_in, embedding], 1)

        dlatents_in = self.g_mapping(latents_in)

        if self.training:
            # Update moving average of W(dlatent).
            # TODO
            if self.truncation is not None:
                self.truncation.update(dlatents_in[0, 0].detach())

            # Realiza la regularización de la mezcla de estilos.
            if self.style_mixing_prob is not None and self.style_mixing_prob > 0:
                latents2 = torch.randn(latents_in.shape).to(latents_in.device)
                dlatents2 = self.g_mapping(latents2)
                layer_idx = torch.from_numpy(np.arange(self.num_layers)[np.newaxis, :, np.newaxis]).to(
                    latents_in.device)
                cur_layers = 2 * (depth + 1)
                mixing_cutoff = random.randint(1,
                                               cur_layers) if random.random() < self.style_mixing_prob else cur_layers
                dlatents_in = torch.where(layer_idx < mixing_cutoff, dlatents_in, dlatents2)

            # Apply truncation trick.
            if self.truncation is not None:
                dlatents_in = self.truncation(dlatents_in)

        fake_images = self.g_synthesis(dlatents_in, depth, alpha)

        return fake_images


class Discriminator(nn.Module):

    def __init__(self, resolution, num_channels=3, conditional=False,
                 n_classes=0, fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 nonlinearity='lrelu', use_wscale=True, mbstd_group_size=4,
                 mbstd_num_features=1, blur_filter=None, structure='linear',
                 **kwargs):
        super(Discriminator, self).__init__()

        if conditional:
            num_channels *= 2
            self.embeddings = []

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.conditional = conditional
        self.mbstd_num_features = mbstd_num_features
        self.mbstd_group_size = mbstd_group_size
        self.structure = structure

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]

        # create the remaining layers
        blocks = []
        from_rgb = []
        for res in range(resolution_log2, 2, -1):
            blocks.append(DiscriminatorBlock(nf(res - 1), nf(res - 2),
                                             gain=gain, activation_layer=nn.LeakyReLU(negative_slope=0.2),
                                             blur_kernel=blur_filter))
            # create the fromRGB layers for various inputs:
            from_rgb.append(Conv2dPropia(num_channels, nf(res - 1), kernel_size=1))
            # Create embeddings for various inputs:
            if conditional:
                r = 2 ** (res)
                self.embeddings.append(
                    Embedding(n_classes, (num_channels // 2) * r * r))

        if self.conditional:
            self.embeddings.append(nn.Embedding( n_classes, (num_channels // 2) * 4 * 4))
            self.embeddings = nn.ModuleList(self.embeddings)

        self.blocks = nn.ModuleList(blocks)

        # Building the final block.
        self.final_block = DiscriminatorTop(self.mbstd_group_size, self.mbstd_num_features,
                                            in_channels=nf(2), intermediate_channels=nf(2),
                                            gain=gain, use_wscale=use_wscale, activation_layer=act)
        from_rgb.append(EqualizedConv2d(num_channels, nf(2), kernel_size=1,
                                        gain=gain, use_wscale=use_wscale))
        self.from_rgb = nn.ModuleList(from_rgb)

        # register the temporary downSampler
        self.temporaryDownsampler = nn.AvgPool2d(2)

    def forward(self, images_in, depth=0, alpha=1., labels_in=None):
        """
        :param images_in: First input: Images [mini_batch, channel, height, width].
        :param labels_in: Second input: Labels [mini_batch, label_size].
        :param depth: current height of operation (Progressive GAN)
        :param alpha: current value of alpha for fade-in
        :return:
        """

        assert depth < self.depth, "Requested output depth cannot be produced"

        if self.conditional:
            assert labels_in is not None, "Conditional Discriminator requires labels"
        # print(embedding_in.shape, images_in.shape)
        # exit(0)
        # print(self.embeddings)
        # exit(0)
        if self.structure == 'fixed':
            if self.conditional:
                embedding_in = self.embeddings[0](labels_in)
                embedding_in = embedding_in.view(images_in.shape[0], -1,
                                                 images_in.shape[2],
                                                 images_in.shape[3])
                images_in = torch.cat([images_in, embedding_in], dim=1)
            x = self.from_rgb[0](images_in)
            for i, block in enumerate(self.blocks):
                x = block(x)
            scores_out = self.final_block(x)

        elif self.structure == 'linear':
            if depth > 0:
                if self.conditional:
                    embedding_in = self.embeddings[self.depth -
                                                   depth - 1](labels_in)
                    embedding_in = embedding_in.view(images_in.shape[0], -1,
                                                     images_in.shape[2],
                                                     images_in.shape[3])
                    images_in = torch.cat([images_in, embedding_in], dim=1)

                residual = self.from_rgb[self.depth -
                                         depth](self.temporaryDownsampler(images_in))
                straight = self.blocks[self.depth - depth -
                                       1](self.from_rgb[self.depth - depth - 1](images_in))
                x = (alpha * straight) + ((1 - alpha) * residual)

                for block in self.blocks[(self.depth - depth):]:
                    x = block(x)
            else:
                if self.conditional:
                    embedding_in = self.embeddings[-1](labels_in)
                    embedding_in = embedding_in.view(images_in.shape[0], -1,
                                                     images_in.shape[2],
                                                     images_in.shape[3])
                    images_in = torch.cat([images_in, embedding_in], dim=1)
                x = self.from_rgb[-1](images_in)

            scores_out = self.final_block(x)
        else:
            raise KeyError("Unknown structure: ", self.structure)

        return scores_out


class StyleGAN:

    def __init__(self, structure, resolution, num_channels, latent_size,
                 g_args, d_args, g_opt_args, d_opt_args, conditional=False,
                 n_classes=0, loss="relativistic-hinge", drift=0.001, d_repeats=1,
                 use_ema=False, ema_decay=0.999, device=torch.device("cpu")):


        # Check conditional validity
        if conditional:
            assert n_classes > 0, "Conditional GANs require n_classes > 0"
        self.structure = structure
        self.depth = int(np.log2(resolution)) - 1
        self.latent_size = latent_size
        self.device = device
        self.d_repeats = d_repeats
        self.conditional = conditional
        self.n_classes = n_classes

        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # Create the Generator and the Discriminator
        self.gen = Generator(num_channels=num_channels,
                             resolution=resolution,
                             structure=self.structure,
                             conditional=self.conditional,
                             n_classes=self.n_classes,
                             **g_args).to(self.device)

        self.dis = Discriminator(num_channels=num_channels,
                                 resolution=resolution,
                                 structure=self.structure,
                                 conditional=self.conditional,
                                 n_classes=self.n_classes,
                                 **d_args).to(self.device)

        # if code is to be run on GPU, we can use DataParallel:
        # TODO

        # define the optimizers for the discriminator and generator
        self.__setup_gen_optim(**g_opt_args)
        self.__setup_dis_optim(**d_opt_args)

        # define the loss function used for training the GAN
        self.drift = drift
        self.loss = self.__setup_loss(loss)

        # Use of ema
        if self.use_ema:
            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)
            # updater function:
            self.ema_updater = update_average
            # initialize the gen_shadow weights equal to the weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

    def __setup_gen_optim(self, learning_rate, beta_1, beta_2, eps):
        self.gen_optim = torch.optim.Adam(self.gen.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps)

    def __setup_dis_optim(self, learning_rate, beta_1, beta_2, eps):
        self.dis_optim = torch.optim.Adam(self.dis.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps)

    def __setup_loss(self, loss):
        if isinstance(loss, str):
            loss = loss.lower()  # lowercase the string

            if not self.conditional:
                assert loss in ["logistic", "hinge", "standard-gan",
                                "relativistic-hinge"], "Unknown loss function"
                if loss == "logistic":
                    loss_func = Losses.LogisticGAN(self.dis)
                elif loss == "hinge":
                    loss_func = Losses.HingeGAN(self.dis)
                if loss == "standard-gan":
                    loss_func = Losses.StandardGAN(self.dis)
                elif loss == "relativistic-hinge":
                    loss_func = Losses.RelativisticAverageHingeGAN(self.dis)
            else:
                assert loss in ["conditional-loss"]
                if loss == "conditional-loss":
                    loss_func = Losses.ConditionalGANLoss(self.dis)

        return loss_func

    def __progressive_down_sampling(self, real_batch, depth, alpha):
        """
        private helper for down_sampling the original images in order to facilitate the
        progressive growing of the layers.

        :param real_batch: batch of real samples
        :param depth: depth at which training is going on
        :param alpha: current value of the fade-in alpha
        :return: real_samples => modified real batch of samples
        """

        from torch.nn import AvgPool2d
        from torch.nn.functional import interpolate

        if self.structure == 'fixed':
            return real_batch

        # down_sample the real_batch for the given depth
        down_sample_factor = int(np.power(2, self.depth - depth - 1))
        prior_down_sample_factor = max(int(np.power(2, self.depth - depth)), 0)

        ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)

        if depth > 0:
            prior_ds_real_samples = interpolate(AvgPool2d(prior_down_sample_factor)(real_batch), scale_factor=2)
        else:
            prior_ds_real_samples = ds_real_samples

        # real samples are a combination of ds_real_samples and prior_ds_real_samples
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

        # return the so computed real_samples
        return real_samples

    def optimize_discriminator(self, noise, real_batch, depth, alpha, labels=None):
        """
        performs one step of weight update on discriminator using the batch of data

        :param noise: input noise of sample generation
        :param real_batch: real samples batch
        :param depth: current depth of optimization
        :param alpha: current alpha for fade-in
        :return: current loss (Wasserstein loss)
        """

        real_samples = self.__progressive_down_sampling(real_batch, depth, alpha)

        loss_val = 0
        for _ in range(self.d_repeats):
            # generate a batch of samples
            fake_samples = self.gen(noise, depth, alpha, labels).detach()

            if not self.conditional:
                loss = self.loss.dis_loss(
                    real_samples, fake_samples, depth, alpha)
            else:
                loss = self.loss.dis_loss(
                    real_samples, fake_samples, labels, depth, alpha)
            # optimize discriminator
            self.dis_optim.zero_grad()
            loss.backward()
            self.dis_optim.step()

            loss_val += loss.item()

        return loss_val / self.d_repeats

    def optimize_generator(self, noise, real_batch, depth, alpha, labels=None):
        """
        performs one step of weight update on generator for the given batch_size

        :param noise: input random noise required for generating samples
        :param real_batch: batch of real samples
        :param depth: depth of the network at which optimization is done
        :param alpha: value of alpha for fade-in effect
        :return: current loss (Wasserstein estimate)
        """

        real_samples = self.__progressive_down_sampling(real_batch, depth, alpha)

        # generate fake samples:
        fake_samples = self.gen(noise, depth, alpha, labels)

        # Change this implementation for making it compatible for relativisticGAN
        if not self.conditional:
            loss = self.loss.gen_loss(real_samples, fake_samples, depth, alpha)
        else:
            loss = self.loss.gen_loss(
                real_samples, fake_samples, labels, depth, alpha)

        # optimize the generator
        self.gen_optim.zero_grad()
        loss.backward()
        # Gradient Clipping
        nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=10.)
        self.gen_optim.step()

        # if use_ema is true, apply ema to the generator parameters
        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)

        # return the loss value
        return loss.item()

    @staticmethod
    def create_grid(samples, scale_factor, img_file):
        """
        utility function to create a grid of GAN samples

        :param samples: generated samples for storing
        :param scale_factor: factor for upscaling the image
        :param img_file: name of file to write
        :return: None (saves a file)
        """
        from torch.nn.functional import interpolate
        from torchvision.utils import save_image

        # upsample the image
        if scale_factor > 1:
            samples = interpolate(samples, scale_factor=scale_factor)

        # save the images:
        save_image(samples, img_file, nrow=int(np.sqrt(len(samples))),
                   normalize=True, scale_each=True, pad_value=128, padding=1)

    def train(self, dataset, num_workers, epochs, batch_sizes, fade_in_percentage, logger, output,
              num_samples=36, start_depth=0, feedback_factor=100, checkpoint_factor=1):
        """
        Utility method for training the GAN. Note that you don't have to necessarily use this
        you can use the optimize_generator and optimize_discriminator for your own training routine.

        :param dataset: object of the dataset used for training.
                        Note that this is not the data loader (we create data loader in this method
                        since the batch_sizes for resolutions can be different)
        :param num_workers: number of workers for reading the data. def=3
        :param epochs: list of number of epochs to train the network for every resolution
        :param batch_sizes: list of batch_sizes for every resolution
        :param fade_in_percentage: list of percentages of epochs per resolution used for fading in the new layer
                                   not used for first resolution, but dummy value still needed.
        :param logger:
        :param output: Output dir for samples,models,and log.
        :param num_samples: number of samples generated in sample_sheet. def=36
        :param start_depth: start training from this depth. def=0
        :param feedback_factor: number of logs per epoch. def=100
        :param checkpoint_factor:
        :return: None (Writes multiple files to disk)
        """

        assert self.depth <= len(epochs), "epochs not compatible with depth"
        assert self.depth <= len(batch_sizes), "batch_sizes not compatible with depth"
        assert self.depth <= len(fade_in_percentage), "fade_in_percentage not compatible with depth"

        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()
        if self.use_ema:
            self.gen_shadow.train()

        # create a global time counter
        global_time = time.time()

        # create fixed_input for debugging
        fixed_input = torch.randn(num_samples, self.latent_size).to(self.device)

        fixed_labels = None
        if self.conditional:
            fixed_labels = torch.linspace(
                0, self.n_classes - 1, num_samples).to(torch.int64).to(self.device)
        # config depend on structure
        logger.info("Starting the training process ... \n")
        if self.structure == 'fixed':
            start_depth = self.depth - 1
        step = 1  # counter for number of iterations
        for current_depth in range(start_depth, self.depth):
            current_res = np.power(2, current_depth + 2)
            logger.info("Currently working on depth: %d", current_depth + 1)
            logger.info("Current resolution: %d x %d" % (current_res, current_res))

            ticker = 1

            # Choose training parameters and configure training ops.
            # TODO
            data = get_data_loader(dataset, batch_sizes[current_depth], num_workers)

            for epoch in range(1, epochs[current_depth] + 1):
                start = timeit.default_timer()  # record time at the start of epoch

                logger.info("Epoch: [%d]" % epoch)
                # total_batches = len(iter(data))
                total_batches = len(data)

                fade_point = int((fade_in_percentage[current_depth] / 100)
                                 * epochs[current_depth] * total_batches)

                for i, batch in enumerate(data, 1):
                    # calculate the alpha for fading in the layers
                    alpha = ticker / fade_point if ticker <= fade_point else 1

                    # extract current batch of data for training
                    if self.conditional:
                        images, labels = batch
                        labels = labels.to(self.device)
                    else:
                        images = batch
                        labels = None

                    images = images.to(self.device)

                    gan_input = torch.randn(images.shape[0], self.latent_size).to(self.device)

                    # optimize the discriminator:
                    dis_loss = self.optimize_discriminator(gan_input, images, current_depth, alpha, labels)

                    # optimize the generator:
                    gen_loss = self.optimize_generator(gan_input, images, current_depth, alpha, labels)

                    # provide a loss feedback
                    if i % int(total_batches / feedback_factor + 1) == 0 or i == 1:
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
                        logger.info(
                            "Elapsed: [%s] Step: %d  Batch: %d  D_Loss: %f  G_Loss: %f"
                            % (elapsed, step, i, dis_loss, gen_loss))

                        # create a grid of samples and save it
                        os.makedirs(os.path.join(output, 'samples'), exist_ok=True)
                        gen_img_file = os.path.join(output, 'samples', "gen_" + str(current_depth)
                                                    + "_" + str(epoch) + "_" + str(i) + ".png")

                        with torch.no_grad():
                            self.create_grid(
                                samples=self.gen(fixed_input, current_depth, alpha,
                                                 labels_in=fixed_labels).detach() if not self.use_ema
                                else self.gen_shadow(fixed_input, current_depth, alpha,
                                                     labels_in=fixed_labels).detach(),
                                scale_factor=int(
                                    np.power(2, self.depth - current_depth - 1)) if self.structure == 'linear' else 1,
                                img_file=gen_img_file,
                            )

                    # increment the alpha ticker and the step
                    ticker += 1
                    step += 1

                elapsed = timeit.default_timer() - start
                elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
                logger.info("Time taken for epoch: %s\n" % elapsed)

                if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == epochs[current_depth]:
                    save_dir = os.path.join(output, 'models')
                    os.makedirs(save_dir, exist_ok=True)
                    gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(current_depth) + "_" + str(epoch) + ".pth")
                    dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(current_depth) + "_" + str(epoch) + ".pth")
                    gen_optim_save_file = os.path.join(
                        save_dir, "GAN_GEN_OPTIM_" + str(current_depth) + "_" + str(epoch) + ".pth")
                    dis_optim_save_file = os.path.join(
                        save_dir, "GAN_DIS_OPTIM_" + str(current_depth) + "_" + str(epoch) + ".pth")

                    torch.save(self.gen.state_dict(), gen_save_file)
                    logger.info("Saving the model to: %s\n" % gen_save_file)
                    torch.save(self.dis.state_dict(), dis_save_file)
                    torch.save(self.gen_optim.state_dict(), gen_optim_save_file)
                    torch.save(self.dis_optim.state_dict(), dis_optim_save_file)

                    # also save the shadow generator if use_ema is True
                    if self.use_ema:
                        gen_shadow_save_file = os.path.join(
                            save_dir, "GAN_GEN_SHADOW_" + str(current_depth) + "_" + str(epoch) + ".pth")
                        torch.save(self.gen_shadow.state_dict(), gen_shadow_save_file)
                        logger.info("Saving the model to: %s\n" % gen_shadow_save_file)

        logger.info('Training completed.\n')


if __name__ == '__main__':
    print('Done.')
