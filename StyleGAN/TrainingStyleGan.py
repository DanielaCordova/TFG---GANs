import copy

import numpy as np
import torch
import numpy as np
import torch
import torch.nn as nn
import copy
import datetime
import os
import random
import time
import timeit
import warnings
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import AvgPool2d
from torch.nn.functional import interpolate

from StyleGAN.Components import update_average, Losses
from StyleGAN.StyleDiscriminator import Discriminator
from StyleGenerador import Generator


def get_data_loader(dataset, batch_size, num_workers):
    """
    generate the data_loader from the given dataset
    :param dataset: dataset for training (Should be a PyTorch dataset)
                    Make sure every item is an Image
    :param batch_size: batch size of the data
    :param num_workers: num of parallel readers
    :return: dl => data_loader for the dataset
    """
    from torch.utils.data import DataLoader

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )

    return dl


class StyleGAN:

    def __init__(self, resolution, num_channels, latent_size,
                 conditional=False,
                 n_classes=0, loss="logistic", drift=0.001, d_repeats=1,
                 use_ema=True, ema_decay=0.999, device=torch.device("cuda")):

        # Check conditional validity
        if conditional:
            assert n_classes > 0, "Conditional GANs require n_classes > 0"

        self.depth = int(np.log2(resolution)) - 1 ##Hasta la profundidad que se puede llegar (desde 4x4 a 128x128)
        self.latent_size = latent_size
        self.device = device
        self.d_repeats = d_repeats
        self.conditional = conditional
        self.n_classes = n_classes
        self.structure = 'linear'
        num_epochs = []

        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # Generator and  Discriminator
        self.gen = Generator(
                             resolution=resolution,
                             conditional=self.conditional,
                             n_classes=self.n_classes).to(self.device)

        self.dis = Discriminator(num_channels=num_channels,
                                 resolution=resolution,

                                 conditional=self.conditional,
                                 n_classes=self.n_classes
                                 ).to(self.device)


        # Optimizers for the discriminator and generator
        self.gen_optim = torch.optim.Adam(self.gen.parameters(), lr=0.003, betas=(0, 0.99), eps=1e-8)
        self.dis_optim = torch.optim.Adam(self.dis.parameters(), lr=0.003, betas=(0, 0.99), eps=1e-8)

        # define the loss function used for training the GAN
        self.drift = drift
        self.loss = self.lossFunction(loss)

        # Use of ema
        if self.use_ema:
            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)
            # updater function:
            self.ema_updater = update_average
            # initialize the gen_shadow weights equal to the weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

    def lossFunction(self, loss):
        if isinstance(loss, str):
            loss = loss.lower()  # lowercase the string

            if not self.conditional:
                if loss == "logistic":
                    loss_func = Losses.LogisticGAN(self.dis)
            else:
                if loss == "conditional-loss":
                    loss_func = Losses.ConditionalGANLoss(self.dis)

        return loss_func

    def train(self, dataset, num_workers, epochs, batch_sizes, logger, output,
              num_samples=36, start_depth=0, feedback_factor=100, checkpoint_factor=1):

        # turn the generator and discriminator into train mode
        self.initial__time = time.time()
        self.gen_loss = []
        self.gen_loss_plot = []
        self.disc_loss = []
        self.disc_loss_plot = []
        self.ejeX = []

        self.step_times = []
        self.num_steps = []
        self.iter = 0
        self.epoch_times = []
        self.num_epochs = []
        self.ejeX = []
        self.act = 0

        self.gen_loss = []
        self.dis_loss = []
        self.gen_plot_loss = []
        self.dis_plot_loss = []
        self.gen.train()
        self.dis.train()
        if self.use_ema:
            self.gen_shadow.train()


        # Global time counter
        global_time = time.time()

        # For debugging
        fixed_input = torch.randn(num_samples, self.latent_size).to(self.device)

        fixed_labels = None
        if self.conditional:
            fixed_labels = torch.linspace(
                0, self.n_classes - 1, num_samples).to(torch.int64).to(self.device)
        # config depend on structure
        logger.info("Starting the training process ... \n")
        #start_depth = self.depth - 1
        step = 1  # counter for number of iterations

        for current_depth in range(start_depth, self.depth):
            current_res = np.power(2, current_depth + 2)
            logger.info("Depth: %d", current_depth + 1)
            logger.info("Resolution: %d x %d" % (current_res, current_res))
            self.depth2 = current_depth
            ticker = 1

            ##Cargamos datos
            data = get_data_loader(dataset, batch_sizes[current_depth], num_workers)

            for epoch in range(1, epochs[current_depth] + 1):
                start = timeit.default_timer()  # record time at the start of epoch
                self.epoch = epoch
                num_epochs = epoch
                logger.info("Epoch: [%d]" % epoch)
                # total_batches = len(iter(data))
                total_batches = len(data)

                fade_point = int((50 / 100)
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
                    self.dl = dis_loss
                    # optimize the generator:
                    gen_loss = self.optimize_generator(gan_input, images, current_depth, alpha, labels)
                    self.gl = gen_loss

                    self.total_batches = total_batches
                    self.feedback_factor = feedback_factor
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
                        self.plot_losses()
                        self.plot_step_time()

                    self.iter = self.iter + 1

                    # increment the alpha ticker and the step
                    ticker += 1
                    step += 1

                elapsed = timeit.default_timer() - start
                elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
                logger.info("Time taken for epoch: %s\n" % elapsed)

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
                self.plot_epoch_time()

        logger.info('Training completed.\n')

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


        ##summary(self.gen, (noise, depth, alpha, labels))
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
    def __progressive_down_sampling(self, real_batch, depth, alpha):
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

    def plot_epoch_time(self):
        epoch_time = (time.time() - self.initial__time)
        self.epoch_times.append(epoch_time)
        self.num_epochs.append(self.epoch)

        title = str(self.epoch_times[-1]) + " seconds taken"

        if self.iter % int(self.total_batches / self.feedback_factor + 1) == 0 or self.iter == 1:
            print(title)

            plt.plot(self.num_epochs, self.epoch_times, label="Epoch Time")

            plt.title(title)
            plt.legend()


            plt.savefig("plot_epoch_time" + str(self.depth2 )  + "_epoch" + str(self.epoch) + "_iter" + str(self.iter)+ '.svg')


            plt.clf()

    def plot_step_time(self):

        step_time = (time.time() - self.initial__time)
        self.step_times.append(step_time)
        self.num_steps.append(self.iter)

        title = str(self.step_times[-1]) + " seconds to take " + str(self.iter) + " steps"

        ##if self.iter % int(self.total_batches / self.feedback_factor + 1) == 0 or self.iter == 1:
        print(title)

        # Time visualization

        plt.plot(self.num_steps, self.step_times, label="Step Time")

        plt.title(title)
        plt.legend()


        plt.savefig("plot_step_time" + str(self.depth2 )  + "_epoch" + str(self.epoch) + "_iter" + str(self.iter)+'.svg')


        plt.clf()

    def plot_losses(self):

        self.disc_loss_plot.append(self.dl)
        self.gen_loss_plot.append(self.gl)



        self.ejeX.append(self.iter)
        arEjeX = np.array(self.ejeX)

       ## if self.iter % int(self.total_batches / self.feedback_factor + 1) == 0 or self.iter == 1:
        title = "Gen Loss: " + str(self.gen_loss_plot[-1]) + " Disc Loss: " + str(self.disc_loss_plot[-1])
        print(title)
        plt.plot(arEjeX, np.array(self.gen_loss_plot), label="Gen Loss")
        plt.plot(arEjeX, np.array(self.disc_loss_plot), label="Disc Loss")

        plt.title(title)
        plt.legend()


        plt.savefig( "plot_losses_depth" + str(self.depth2 )  + "_epoch" + str(self.epoch) + "_iter" + str(self.iter) + '.svg')


        plt.clf()