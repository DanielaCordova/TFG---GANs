import time
from datetime import datetime, date, time, timedelta
import timeit
from abc import abstractclassmethod
from venv import create
import tqdm
from torch.utils.data import DataLoader

import Constants
import torch
import ImageFunctions
import matplotlib.pyplot as plt
import os

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy


from StyleGAN.Components import update_average, Losses
from StyleGAN.Components.data import get_data_loader




def graph_GANS_losses(x, ejeX, title, dir, save, show):

    for ayval , labeld in x :
        plt.plot(np.array(ejeX), np.array(ayval), label=labeld)

    plt.title(title)
    plt.legend()

    if save:
        os.chdir(dir)
        plt.savefig(datetime.now().strftime("%H-%M-%S-%f %d-%m-%y")+'.png')
        os.chdir('..')
    
    if show: 
        plt.show()

    plt.clf()

def saveCheckpoint(dir, gen, disc, gen_opt, disc_opt, gen_loss, disc_loss, epoch, iter):
    os.chdir(dir)

    g_s = 'gen_' + str(iter) + '.tar'
    d_s = 'disc_' + str(iter) + '.tar'

    torch.save({
       'epoch' : epoch ,
       'iter' : iter,
       'model_state_dict' : gen.state_dict(), 
       'optimizer_state_dict' : gen_opt.state_dict(),
       'loss' : gen_loss
    }, g_s)

    torch.save({
        'epoch' : epoch,
        'iter' : iter,
        'model_state_dict' : disc.state_dict(),
        'optimizer_state_dict' : disc_opt.state_dict(),
        'loss' : disc_loss
    }, d_s)

    os.chdir('..')

class GAN_Trainer:
    def __init__(self, dataloader, generator, discriminator, criterion, log_step, log_dir, checksave = False, save_step = None, load = False, 
    load_dir = None, gen_load = None, disc_load = None, time_steps = True, time_epochs = True, device = 'cuda'):
        self.device        = device
        self.dataloader    = dataloader
        self.generator     = generator
        self.discriminator = discriminator
        self.criterion     = criterion
        self.log_step      = log_step
        self.log_dir       = log_dir
        self.checksave     = checksave
        self.time_epochs   = time_epochs
        self.time_steps    = time_steps
        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=Constants.LR)
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=Constants.LR)
        if checksave :
            self.save_step = save_step
        self.load          = load
        if load:
            self.load_dir  = load_dir
            self.gen_load  = gen_load
            self.disc_load = disc_load

        self.gen_loss      = []
        self.dis_loss      = []
        self.gen_plot_loss = []
        self.dis_plot_loss = []
        self.ejeX          = []
        self.act           = 0
        self.iter          = 0
        if time_steps :
            self.step_times    = []
            self.num_steps     = []
        if time_epochs :
            self.epoch_times   = []
            self.num_epochs    = []

    @abstractclassmethod
    def preprocessRealData(self,real):
        pass

    @abstractclassmethod
    def appendDiscLoss(self, loss):
        pass

    @abstractclassmethod
    def check_per_batch(self, real, fake, it):
        pass

    def epoch(self, ep):
        it = 0
        for real in tqdm(self.dataloader):
            if real[0].shape[0] != Constants.BATCH_SIZE :
                print(real[0].shape[0])
                break

            real = self.preprocessRealData(real)

            g_loss, generated = self.train_gen(real)
            t_loss = self.train_disc(real)

            self.gen_loss.append(g_loss)
            self.appendDiscLoss(t_loss)

            self.iter = self.iter + 1
            it = it + 1

            if self.iter % self.log_step == 0 and self.iter > 0 :
                self.plot_losses()
                self.save_results(real, generated)
                
                if self.time_steps:
                    self.plot_step_time()
        
            if self.checksave and self.iter % self.save_step == 0:
                self.saveCheckpoint(self.log_dir, self.generator, self.discriminator, self.gen_opt, self.disc_opt, self.gen_loss[-1], self.dis_loss[-1], ep)

            self.check_per_batch(real, generated, it)
        
        if self.time_epochs:
            self.plot_epoch_time()

    @abstractclassmethod
    def train_gen(self, real_data):
        pass

    @abstractclassmethod
    def train_disc(self, real_data):
        pass

    @abstractclassmethod
    def train_for_epochs(self, n_epochs):
        pass

    @abstractclassmethod
    def load_checkpoint(self):
        pass

    @abstractclassmethod
    def plot_losses(self):
        pass

    def save_results(self, real, generated):
        ImageFunctions.tensor_as_image(generated, self.iter, "fake", self.log_dir, save = True, show = False)
        ImageFunctions.tensor_as_image(real, self.iter, "real", self.log_dir, save = True, show = False) 

    @abstractclassmethod
    def saveCheckpoint(self):
        pass

    def enable_training(self, model, flag):
        for p in model.parameters():
            p.requires_grad = flag

    def check_training_params(self, model, flag):
        for p in model.parameters():
            assert(p.requires_grad == flag)
        
    def plot_epoch_time(self):
        epoch_time = (time.time() - self.initial__time)
        self.epoch_times.append(epoch_time)
        self.num_epochs.append(self.act)

        title = str(self.epoch_times[-1]) + " seconds taken"
        print(title)

        plt.plot(self.num_epochs,self.epoch_times, label="Epoch Time")
        
        plt.title(title)
        plt.legend()

        os.chdir(self.log_dir)
        plt.savefig(datetime.now().strftime("%d-%m")+" iter " + str(self.act) + " epoch_times" + '.pdf')
        os.chdir('..')

        plt.clf()
        
    def plot_step_time(self):
        
        step_time = (time.time() - self.initial__time)
        self.step_times.append(step_time)
        self.num_steps.append(self.iter)

        title = str(self.step_times[-1]) + " seconds to take " + str(self.iter) + " steps"
        print(title)
        
        #Time visualization

        plt.plot(self.num_steps, self.step_times, label="Step Time")
        
        plt.title(title)
        plt.legend()

        os.chdir(self.log_dir)
        plt.savefig(datetime.now().strftime("%d-%m")+" iter " + str(self.iter) + " step_times" + '.pdf')
        os.chdir('..')

        plt.clf()
        
        
class Normal_Trainer(GAN_Trainer):
    def __init__(self, dataloader, generator, discriminator, criterion, log_step, log_dir, device = 'cuda', verb = False, checksave = False, save_step = None, load = False, load_dir = None, gen_load = None, disc_load = None, time_steps = False, time_epochs = False):

        super().__init__(dataloader, generator, discriminator, criterion, log_step, log_dir, checksave, save_step, load, load_dir, gen_load, disc_load, time_steps, time_epochs, device = device)
        self.verb          = verb

        if verb :
            self.dis_fake_loss_plot = []
            self.dis_real_loss_plot = []
            self.dis_real_loss      = []
            self.dis_fake_loss      = []
        
        if load :
            self.load_checkpoint()
    
    def load_checkpoint(self):
        os.chdir(self.load_dir)

        c_g = torch.load(self.gen_load)
        c_d = torch.load(self.disc_load)

        self.act = c_d['epoch']

        self.discriminator.load_state_dict(c_d['model_state_dict'])
        self.disc_opt.load_state_dict(c_d['optimizer_state_dict'])
        self.dis_loss.append( c_d['loss'] )


        self.generator.load_state_dict(c_g['model_state_dict'])
        self.gen_opt.load_state_dict(c_g['optimizer_state_dict'])
        self.gen_loss.append( c_g['loss'] )

        os.chdir('..')

    def saveCheckpoint(self, epoch):
        os.chdir(self.log_dir)

        g_s = 'gen_' + str(epoch) + '.tar'
        d_s = 'disc_' + str(epoch) + '.tar'

        torch.save({
        'epoch' : epoch,
        'model_state_dict' : self.generator.state_dict(), 
        'optimizer_state_dict' : self.gen_opt.state_dict(),
        'loss' : self.gen_loss[-1]
        }, g_s)

        torch.save({
            'epoch' : epoch,
            'model_state_dict' : self.discriminator.state_dict(),
            'optimizer_state_dict' : self.disc_opt.state_dict(),
            'loss' : self.dis_loss[-1]
        }, d_s)

        os.chdir('..')

    def appendDiscLoss(self, discLoss):
        
        if self.verb:
            total_loss , real_loss , fake_loss = discLoss
            self.dis_real_loss.append(real_loss)
            self.dis_fake_loss.append(fake_loss)
            self.dis_loss.append(total_loss)
        else:
            self.dis_loss.append(total_loss)

    def check_per_batch(self, real, generated, it):
        return
    
    def train_gen(self, real):
        self.enable_training(self.generator, True)
        self.enable_training(self.discriminator, False)


        real = real.to(self.device)
        noise = torch.randn((Constants.BATCH_SIZE, self.generator.getNoiseDim())).to(self.device)

        fake = self.generator(noise)

        f_pred = self.discriminator(fake)
        
        loss = self.criterion(f_pred, torch.ones_like(f_pred))

        self.gen_opt.zero_grad()
        loss.backward()
        self.gen_opt.step()

        return (loss.item(), fake)
    
    def train_disc(self, real):
        self.enable_training(self.generator, False)
        self.enable_training(self.discriminator, True)

        real = real.to(self.device)
        noise = torch.randn((Constants.BATCH_SIZE, self.generator.getNoiseDim())).to(self.device)

        fake = self.generator(noise)

        f_pred = self.discriminator(fake)
        r_pred = self.discriminator(real)

        fake_loss = self.criterion(f_pred, torch.zeros_like(f_pred))
        real_loss = self.criterion(r_pred, torch.ones_like(r_pred))
        total_loss = (fake_loss + real_loss) / 2

        self.disc_opt.zero_grad()
        total_loss.backward()
        self.disc_opt.step()

        return total_loss.item(), real_loss.item(), fake_loss.item()

    def train_for_epochs(self, n_epochs):
        self.initial__time = time.time()
        print(self.initial__time)
        for e in range(0, n_epochs):
            self.epoch(e + self.act)
            self.act += 1

    def save_results(self, real, generated):
        ImageFunctions.tensor_as_image_gray(generated, self.iter, "fake", self.log_dir, save = True, show = False)
        ImageFunctions.tensor_as_image_gray(real, self.iter, "real", self.log_dir, save = True, show = False)

    def epoch(self, ep):
        it = 0
        for real in tqdm(self.dataloader):
            if real[0].shape[0] != Constants.BATCH_SIZE :
                print(real[0].shape[0])
                break

            real, tag = real

            g_loss, generated = self.train_gen(real)
            t_loss = self.train_disc(real)

            self.gen_loss.append(g_loss)
            self.appendDiscLoss(t_loss)

            self.iter = self.iter + 1
            it = it + 1

            if self.iter % self.log_step == 0 and self.iter > 0 :
                self.plot_losses()
                self.save_results(real, generated)
            
                if self.time_steps:
                    self.plot_step_time()
        
            if self.checksave and self.iter % self.save_step == 0:
                self.saveCheckpoint(ep)
        
        
        if self.time_epochs:
            self.plot_epoch_time()

        self.check_per_batch(real, generated, it)
    
    def plot_losses(self):
        
        self.dis_plot_loss.append(sum(self.dis_loss[-self.log_step:])/self.log_step)
        self.gen_plot_loss.append(sum(self.gen_loss[-self.log_step:])/self.log_step)

        if self.verb :
            l1 = sum(self.dis_fake_loss[-self.log_step:])/self.log_step
            l2 = sum(self.dis_real_loss[-self.log_step:])/self.log_step
            self.dis_fake_loss_plot.append(l1)
            self.dis_real_loss_plot.append(l2)

        title = "Gen Loss: " + str(self.gen_plot_loss[-1]) + " Disc Loss: " + str(self.dis_plot_loss[-1])
        print(title)
        
        self.ejeX.append(self.iter)
        arEjeX = np.array(self.ejeX)
        plt.plot(arEjeX, np.array(self.gen_plot_loss), label = "Gen Loss")
        plt.plot(arEjeX, np.array(self.dis_plot_loss), label = "Disc Loss")

        if self.verb :
            plt.plot(arEjeX, np.array(self.dis_fake_loss_plot), label = "Disc Fake Loss")
            plt.plot(arEjeX, np.array(self.dis_real_loss_plot), label = "Disc Real Loss")

        plt.title(title)
        plt.legend()

        os.chdir(self.log_dir)
        plt.savefig(datetime.now().strftime("%d-%m")+" iter " + str(self.iter) + '.pdf')
        os.chdir('..')

        plt.clf()

    def generate_samples(self, n_samples):
        for i in range(0, n_samples):
            noise = torch.randn((Constants.BATCH_SIZE, self.generator.getNoiseDim())).to(self.device)
            samples = self.generator(noise)
            ImageFunctions.tensor_as_image_gray(samples, self.iter, "sample", self.log_dir, save = True, show = False)
        
        

class Cond_Trainer(GAN_Trainer):
    def __init__(self, dataloader, generator, discriminator, criterion, log_step, log_dir, num_classes, device = 'cuda', verb = False, checksave = False, save_step = None, load = False, load_dir = None, gen_load = None, disc_load = None, time_steps = False, time_epochs = False):

        super().__init__(dataloader, generator, discriminator, criterion, log_step, log_dir, checksave, save_step, load, load_dir, gen_load, disc_load, time_steps, time_epochs, device = device)
        self.verb          = verb
        self.num_classes   = num_classes

        if verb :
            self.dis_fake_loss_plot = []
            self.dis_real_loss_plot = []
            self.dis_real_loss      = []
            self.dis_fake_loss      = []
        
        if load :
            self.load_checkpoint()
    
    def load_checkpoint(self):
        os.chdir(self.load_dir)

        c_g = torch.load(self.gen_load)
        c_d = torch.load(self.disc_load)

        self.act = c_d['epoch']

        self.discriminator.load_state_dict(c_d['model_state_dict'])
        self.disc_opt.load_state_dict(c_d['optimizer_state_dict'])
        self.dis_loss.append( c_d['loss'] )


        self.generator.load_state_dict(c_g['model_state_dict'])
        self.gen_opt.load_state_dict(c_g['optimizer_state_dict'])
        self.gen_loss.append( c_g['loss'] )

        os.chdir('..')

    def saveCheckpoint(self, epoch):
        os.chdir(self.log_dir)

        g_s = 'gen_' + str(epoch) + '.tar'
        d_s = 'disc_' + str(epoch) + '.tar'

        torch.save({
        'epoch' : epoch,
        'model_state_dict' : self.generator.state_dict(), 
        'optimizer_state_dict' : self.gen_opt.state_dict(),
        'loss' : self.gen_loss[-1]
        }, g_s)

        torch.save({
            'epoch' : epoch,
            'model_state_dict' : self.discriminator.state_dict(),
            'optimizer_state_dict' : self.disc_opt.state_dict(),
            'loss' : self.dis_loss[-1]
        }, d_s)

        os.chdir('..')
    
    def preprocessRealData(self, real_data):
        real, tag = real_data

        tag = torch.nn.functional.one_hot(tag, self.num_classes)
        img_vec_tag = tag[:,:,None,None]
        img_vec_tag = img_vec_tag.repeat(1,1, real.shape[2], real.shape[3])
        img_vec_tag = torch.cat((real.float(), img_vec_tag.float()),1)

        return img_vec_tag, tag

    def appendDiscLoss(self, discLoss):
        
        if self.verb:
            total_loss , real_loss , fake_loss = discLoss
            self.dis_real_loss.append(real_loss)
            self.dis_fake_loss.append(fake_loss)
            self.dis_loss.append(total_loss)
        else:
            self.dis_loss.append(total_loss)

    def check_per_batch(self, real, generated, it):
        return

    def get_InputVector_paraEtiquetar(self, etiquetas, numClases):
        return F.one_hot(etiquetas,numClases) 

    def combinarVectores(self, x, y):
        return torch.cat((x.float(),y.float()), 1)
    
    def train_gen(self, real):
        self.enable_training(self.generator, True)
        self.enable_training(self.discriminator, False)

        real, tag = real


        real = real.to(self.device)
        tag  = tag.to(self.device)
        noise = torch.randn((Constants.BATCH_SIZE, self.generator.getNoiseDim())).to(self.device)

        noise_tag = self.combinarVectores(noise, tag)

        fake = self.generator(noise_tag)

        img_vec_tag = tag[:,:,None,None]
        img_vec_tag = img_vec_tag.repeat(1,1,fake.shape[2], fake.shape[3])
        fake_tag = torch.cat((fake.float(), img_vec_tag.float()),1)
        f_pred = self.discriminator(fake_tag)
        
        loss = self.criterion(f_pred, torch.ones_like(f_pred))

        self.gen_opt.zero_grad()
        loss.backward()
        self.gen_opt.step()

        return (loss.item(), fake)
    
    def train_disc(self, real):
        self.enable_training(self.generator, False)
        self.enable_training(self.discriminator, True)

        real, tag = real

        real = real.to(self.device)
        tag  = tag.to(self.device)
        noise = torch.randn((Constants.BATCH_SIZE, self.generator.getNoiseDim())).to(self.device)

        noise_tag = self.combinarVectores(noise, tag)

        fake = self.generator(noise_tag)

        img_vec_tag = tag[:,:,None,None]
        img_vec_tag = img_vec_tag.repeat(1,1,fake.shape[2], fake.shape[3])
        fake_tag = torch.cat((fake.float(), img_vec_tag.float()),1)
        f_pred = self.discriminator(fake_tag)

        fake_tag = self.combinarVectores(fake, img_vec_tag) 

        f_pred = self.discriminator(fake_tag)
        r_pred = self.discriminator(real)

        fake_loss = self.criterion(f_pred, torch.zeros_like(f_pred))
        real_loss = self.criterion(r_pred, torch.ones_like(r_pred))
        total_loss = (fake_loss + real_loss) / 2

        self.disc_opt.zero_grad()
        total_loss.backward()
        self.disc_opt.step()

        return total_loss.item(), real_loss.item(), fake_loss.item()

    def train_for_epochs(self, n_epochs):
        self.initial__time = time.time()
        for e in range(0, n_epochs):
            self.epoch(e + self.act)
            self.act += 1

    def epoch(self, ep):
        it = 0
        for real in tqdm(self.dataloader):
            if real[0].shape[0] != Constants.BATCH_SIZE :
                print(real[0].shape[0])
                break
            
            real2 = real
            real = self.preprocessRealData(real)

            g_loss, generated = self.train_gen(real)
            t_loss = self.train_disc(real)

            self.gen_loss.append(g_loss)
            self.appendDiscLoss(t_loss)

            self.iter = self.iter + 1
            it = it + 1

            real, tag = real2
            if self.iter % self.log_step == 0 and self.iter > 0 :
                self.plot_losses()
                self.save_results(real, generated)
            
                if self.time_steps:
                    self.plot_step_time()
        
            if self.checksave and self.iter % self.save_step == 0:
                self.saveCheckpoint(ep)
        
        
        if self.time_epochs:
            self.plot_epoch_time()

        self.check_per_batch(real, generated, it)
    
    def plot_losses(self):
        
        self.dis_plot_loss.append(sum(self.dis_loss[-self.log_step:])/self.log_step)
        self.gen_plot_loss.append(sum(self.gen_loss[-self.log_step:])/self.log_step)

        if self.verb :
            l1 = sum(self.dis_fake_loss[-self.log_step:])/self.log_step
            l2 = sum(self.dis_real_loss[-self.log_step:])/self.log_step
            self.dis_fake_loss_plot.append(l1)
            self.dis_real_loss_plot.append(l2)

        title = "Gen Loss: " + str(self.gen_plot_loss[-1]) + " Disc Loss: " + str(self.dis_plot_loss[-1])
        print(title)
        
        self.ejeX.append(self.iter)
        arEjeX = np.array(self.ejeX)
        plt.plot(arEjeX, np.array(self.gen_plot_loss), label = "Gen Loss")
        plt.plot(arEjeX, np.array(self.dis_plot_loss), label = "Disc Loss")

        if self.verb :
            plt.plot(arEjeX, np.array(self.dis_fake_loss_plot), label = "Disc Fake Loss")
            plt.plot(arEjeX, np.array(self.dis_real_loss_plot), label = "Disc Real Loss")

        plt.title(title)
        plt.legend()

        os.chdir(self.log_dir)
        plt.savefig(datetime.now().strftime("%d-%m")+" iter " + str(self.iter) + '.pdf')
        os.chdir('..')

        plt.clf()

    def generate_samples(self, num_samples):
        for i in range(0, num_samples):
            noise = torch.randn((1, self.generator.getNoiseDim())).to(self.device)
            tag = torch.randint(0, self.num_classes, (1, 1)).to(self.device)
            oh_tag = torch.nn.functional.one_hot(tag, self.num_classes)
            oh_tag = oh_tag.view(1, -1)
            noise_tag = self.combinarVectores(noise, oh_tag)
            fake = self.generator(noise_tag)
            ImageFunctions.tensor_as_image(fake, i, "sample", self.log_dir, save = True, show = False)
            
        

class Style_Prog_Trainer:
    def __init__(self, generator, discriminator, resolution, num_channels, latent_size,
                 conditional=False,
                 n_classes=0, loss="logistic", drift=0.001, d_repeats=1,
                 use_ema=True, ema_decay=0.999, device=torch.device("cuda"),
                    checksave = False, load = False, load_dir = None, gen_load = None, disc_load = None, time_steps = True, time_epochs = True,
                 ):
        # Check conditional validity
        if conditional:
            assert n_classes > 0, "Conditional GANs require n_classes > 0"

        self.depth = int(np.log2(resolution)) - 1  ##Hasta la profundidad que se puede llegar (desde 4x4 a 128x128)
        self.latent_size = latent_size
        self.device = device
        self.d_repeats = d_repeats
        self.conditional = conditional
        self.n_classes = n_classes
        self.structure = 'linear'
        num_epochs = []
        self.checksave=checksave
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # Generator and  Discriminator
        self.gen = generator(
            resolution=resolution,
            conditional=self.conditional,
            n_classes=self.n_classes).to(self.device)

        self.dis = discriminator(num_channels=num_channels,
                                 resolution=resolution,

                                 conditional=self.conditional,
                                 n_classes=self.n_classes
                                 ).to(self.device)


        self.device = device
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


        self.resdir = dir
        self.log_step = 0

        self.iter = 0

        # Optimizers for the discriminator and generator
        self.gen_optim = torch.optim.Adam(self.gen.parameters(), lr=0.003, betas=(0, 0.99), eps=1e-8)
        self.dis_optim = torch.optim.Adam(self.dis.parameters(), lr=0.003, betas=(0, 0.99), eps=1e-8)
        self.downsampler = nn.AvgPool2d(4,2,1)


        self.gen_loss = []
        self.gen_loss_plot = []
        self.disc_loss = []
        self.disc_loss_plot = []
        self.ejeX = []

        self.act_epoch = 0
        
        if time_steps :
            self.step_times    = []
            self.num_steps     = []
        if time_epochs :
            self.epoch_times   = []
            self.num_epochs    = []

        if load :
            os.chdir(load_dir)

            c_g = torch.load(gen_load)
            c_d = torch.load(disc_load)

            self.ac = c_d['epoch']

            self.disc.load_state_dict(c_d['model_state_dict'])
            self.disc_opt.load_state_dict(c_d['optimizer_state_dict'])
            self.disc_loss.append( c_d['loss'] )
            self.disc.alfa = c_d['alfa']
            self.disc.depth = c_d['depth']

            self.gen.load_state_dict(c_g['model_state_dict'])
            self.gen_opt.load_state_dict(c_g['optimizer_state_dict'])
            self.gen_loss.append( c_g['loss'] )
            self.gen.alfa = c_g['alfa']
            self.gen.depth = c_g['depth']

            os.chdir('..')

        if checksave:
            self.save_step = 0


        else:
            self.prog = False
            self.alfa_step = 0
            self.increase_alfa_step = 0
        # Check conditional validity
        if conditional:
            assert n_classes > 0, "Conditional GANs require n_classes > 0"

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

    def enable_training(self, model, flag):
        for p in model.parameters():
            p.requires_grad = flag

    def check_training_params(self, model, flag):
        for p in model.parameters():
            assert(p.requires_grad == flag)

    def train_gen(self, real_data):
        
        in_size = self.disc.getinSize()
        while real_data.shape[2] != in_size:
            real_data = self.downsampler(real_data)

        self.enable_training(self.gen, True)
        self.enable_training(self.disc,False)

        noise = torch.randn(Constants.BATCH_SIZE, 512).to(self.device)

        fake = self.gen(noise)
        fake_pred = self.disc(fake)
        real_pred = self.disc(real_data)

        loss = self.criterion.gen_loss(real_data, fake, real_pred, fake_pred)

        self.check_training_params(self.gen, True)
        self.check_training_params(self.disc, False)

        self.gen_opt.zero_grad()
        loss.backward(retain_graph = True)
        self.gen_opt.step()
        
        self.check_training_params(self.gen, True)
        self.check_training_params(self.disc, False)

        return (loss.item(), fake)


    def train_disc(self, real_data):
        
        real_data = real_data.to(self.device)


        self.enable_training(self.gen, False)
        self.enable_training(self.disc,True)

        noise = torch.randn(Constants.BATCH_SIZE, 512).to('cuda')

        fake = self.gen(noise)
        fake_pred = self.disc(fake)

        real_pred = self.disc(real_data)

        perdidasTotales = self.criterion.dis_loss(real_data, fake,real_pred, fake_pred)

        self.disc_opt.zero_grad()
        perdidasTotales.backward()
        self.disc_opt.step()

        return perdidasTotales.item()

    def epoch(self, ep):
        it = 0
        for real in tqdm(self.dataloader):
            if real.shape[0] !=  self.batch_size:
                break

            real = real.to(self.device)

            in_size = self.disc.getinSize()
            while real.shape[2] != in_size:
                real = self.downsampler(real)

            g_loss, generated = self.train_gen(real)
            t_loss = self.train_disc(real)

            self.gen_loss.append(g_loss)
            self.disc_loss.append(t_loss)

            self.iter = self.iter + 1
            it = it + 1

            if self.iter % self.log_step == 0 and self.iter > 0 :
                self.plot_losses()
                self.save_results(real, generated)

                self.plot_step_time()

            if self.iter % self.increase_alfa_step == 0 and self.iter > 0:
                self.gen.increaseAlfa(self.alfa_step)
                self.disc.increaseAlfa(self.alfa_step)
        
            if self.iter % self.save_step == 0 and self.iter > 0:
                self.saveCheckpoint(ep)
                
        self.plot_epoch_time()


    def train_for_epochs(self, dataset, num_workers, epochs, batch_sizes, logger, output,
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
            self.fixed_labels = torch.linspace(
                0, self.n_classes - 1, num_samples).to(torch.int64).to(self.device)
        # config depend on structure
        logger.info("Starting the training process ... \n")
        # start_depth = self.depth - 1
        step = 1  # counter for number of iterations

        for current_depth in range(start_depth, self.depth):   ##Profundidad actual

            current_res = np.power(2, current_depth + 2)
            logger.info("Depth: %d", current_depth + 1)
            logger.info("Resolution: %d x %d" % (current_res, current_res))

            logger.info("      Epochs for depth {} = {}".format(current_depth, epochs[current_depth]))
            logger.info("      Batch size for depth {} = {}".format(current_depth, batch_sizes[current_depth]))
            #logger.info("      Increase_alfa_step for depth {} = {}".format(current_depth, self.increase_alfa_step))
            #logger.info("      Alfa step for depth {} = {}".format(current_depth, self.alfa_step))
            self.depth2 = current_depth
            ticker = 1

            ##Cargamos datos
            data = get_data_loader(dataset, batch_sizes[current_depth], num_workers)
            self.num_ep=0
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
                    self.alpha = ticker / fade_point if ticker <= fade_point else 1

                    # extract current batch of data for training
                    if self.conditional:
                        images, labels = batch
                        labels = labels.to(self.device)
                    else:
                        images = batch
                        labels = None

                    images = images.to(self.device)
                    self.inputShape=images.shape
                    gan_input = torch.randn(images.shape[0], self.latent_size).to(self.device)

                    # optimize the discriminator:
                    dis_loss = self.optimize_discriminator(gan_input, images, current_depth,  self.alpha, labels)
                    self.dl = dis_loss
                    # optimize the generator:
                    gen_loss = self.optimize_generator(gan_input, images, current_depth,  self.alpha, labels)
                    self.gl = gen_loss

                    self.total_batches = total_batches
                    self.feedback_factor = feedback_factor
                    # provide a loss feedback

                    if i % int(total_batches / feedback_factor + 1) == 0 or i == 1:
                        elapsed = time.time() - global_time
                        elapsed = str(timedelta(seconds=elapsed)).split('.')[0]
                        logger.info(
                            "Elapsed: [%s] Step: %d  Batch: %d  D_Loss: %f  G_Loss: %f"
                            % (elapsed, step, i, dis_loss, gen_loss))

                        # create a grid of samples and save it
                        os.makedirs(os.path.join(output, 'samples'), exist_ok=True)
                        gen_img_file = os.path.join(output, 'samples', "gen_" + str(current_depth)
                                                    + "_" + str(epoch) + "_" + str(i) + ".png")

                        if self.checksave:
                            self.saveCheckpoint(epoch)
                        with torch.no_grad():
                            self.create_grid(
                                samples=self.gen(fixed_input, current_depth,  self.alpha,
                                                 labels_in=fixed_labels).detach() if not self.use_ema
                                else self.gen_shadow(fixed_input, current_depth,  self.alpha,
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
                logger.info(" ## Training data for size {} ".format(self.inputShape))
                elapsed = timeit.default_timer() - start
                elapsed = str(timedelta(seconds=elapsed)).split('.')[0]
                logger.info("Time taken for epoch: %s\n" % elapsed)
                self.num_ep = self.num_ep + 1

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

    def optimize_discriminator(self, noise, real_batch, depth, alpha, labels=None):
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


        real_samples = self.__progressive_down_sampling(real_batch, depth, alpha)

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

        ds_real_samples = nn.AvgPool2d(down_sample_factor)(real_batch)

        if depth > 0:
            prior_ds_real_samples = F.interpolate(nn.AvgPool2d(prior_down_sample_factor)(real_batch), scale_factor=2)
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

            plt.savefig(
                "plot_epoch_time" + str(self.depth2) + "_epoch" + str(self.epoch) + "_iter" + str(self.iter) + '.svg')

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

        plt.savefig(
            "plot_step_time" + str(self.depth2) + "_epoch" + str(self.epoch) + "_iter" + str(self.iter) + '.svg')

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

        plt.savefig(
            "plot_losses_depth" + str(self.depth2) + "_epoch" + str(self.epoch) + "_iter" + str(self.iter) + '.svg')

        plt.clf()
    

    def save_results(self, real, generated):
        ImageFunctions.tensor_as_image(generated, self.iter, "fake", self.resdir, save = True, show = False)
        ImageFunctions.tensor_as_image(real, self.iter, "real", self.resdir, save = True, show = False) 
          

    def saveCheckpoint(self, epoch):
        os.chdir(self.resdir)

        g_s = 'gen_' + str(epoch) + '.tar'
        d_s = 'disc_' + str(epoch) + '.tar'

        torch.save({
        'epoch' : epoch,
        'alfa' : self.alpha,
        'depth': self.depth,
        'model_state_dict' : self.gen.state_dict(), 
        'optimizer_state_dict' : self.gen_opt.state_dict(),
        'loss' : self.gen_loss[-1]
        }, g_s)

        torch.save({
            'epoch' : epoch,
            'alfa' : self.alpha,
            'depth': self.depth,
            'model_state_dict' : self.disc.state_dict(),
            'optimizer_state_dict' : self.disc_opt.state_dict(),
            'loss' : self.disc_loss[-1]
        }, d_s)

        os.chdir('..')
        




class Cycle_Trainer():
    def __init__(self, dataloader1, dataloader2, gen_AB, gen_BA, gen_opt, disc_A, disc_A_opt, disc_B, disc_B_opt,
                 adv_criterion,
                 recon_criterion, log_step, log_dir, target_shape, device='cuda', checksave=False, save_step=None,
                 load=False,
                 load_dir=None, gen_disc_load=None, time_steps=False, time_epochs=False):

        self.device = device
        self.adv_criterion = adv_criterion
        self.recon_criterion = recon_criterion
        self.log_step = log_step
        self.log_dir = log_dir
        self.checksave = checksave
        self.time_epochs = time_epochs
        self.time_steps = time_steps
        self.target_shape = target_shape

        if checksave:
            self.save_step = save_step

        self.load = load

        if load:
            self.load_dir = load_dir
            self.gen_disc_load = gen_disc_load

        self.act = 0
        self.iter = 0

        if time_steps:
            self.step_times = []
            self.num_steps = []
        if time_epochs:
            self.epoch_times = []
            self.num_epochs = []

        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2

        self.gen_AB = gen_AB
        self.gen_BA = gen_BA
        self.gen_opt = gen_opt
        self.disc_A = disc_A
        self.disc_A_opt = disc_A_opt
        self.disc_B = disc_B
        self.disc_B_opt = disc_B_opt
        self.gen_loss = []
        self.dis_loss = []

        if load:
            self.load_checkpoint()
        else:
            gen_AB = gen_AB.apply(self.weights_init)
            gen_BA = gen_BA.apply(self.weights_init)
            disc_A = disc_A.apply(self.weights_init)
            disc_B = disc_B.apply(self.weights_init)

    def get_disc_loss(self, real_X, fake_X, disc_X, adv_criterion):
        '''
        Return the loss of the discriminator given inputs.
        Parameters:
            real_X: the real images from pile X
            fake_X: the generated images of class X
            disc_X: the discriminator for class X; takes images and returns real/fake class X
                prediction matrices
            adv_criterion: the adversarial loss function; takes the discriminator
                predictions and the target labels and returns a adversarial
                loss (which you aim to minimize)
        '''
        #### START CODE HERE ####
        disc_fake_X_hat = disc_X(fake_X.detach())  # Detach generator
        disc_fake_X_loss = adv_criterion(disc_fake_X_hat, torch.zeros_like(disc_fake_X_hat))
        disc_real_X_hat = disc_X(real_X)
        disc_real_X_loss = adv_criterion(disc_real_X_hat, torch.ones_like(disc_real_X_hat))
        disc_loss = (disc_fake_X_loss + disc_real_X_loss) / 2
        #### END CODE HERE ####
        return disc_loss

    def get_gen_adversarial_loss(self, real_X, disc_Y, gen_XY, adv_criterion):
        '''
        Return the adversarial loss of the generator given inputs
        (and the generated images for testing purposes).
        Parameters:
            real_X: the real images from pile X
            disc_Y: the discriminator for class Y; takes images and returns real/fake class Y
                prediction matrices
            gen_XY: the generator for class X to Y; takes images and returns the images
                transformed to class Y
            adv_criterion: the adversarial loss function; takes the discriminator
                      predictions and the target labels and returns a adversarial
                      loss (which you aim to minimize)
        '''
        #### START CODE HERE ####
        fake_Y = gen_XY(real_X)
        disc_fake_Y_hat = disc_Y(fake_Y)
        adversarial_loss = adv_criterion(disc_fake_Y_hat, torch.ones_like(disc_fake_Y_hat))
        #### END CODE HERE ####
        return adversarial_loss, fake_Y

    def get_identity_loss(self, real_X, gen_YX, identity_criterion):
        '''
        Return the identity loss of the generator given inputs
        (and the generated images for testing purposes).
        Parameters:
            real_X: the real images from pile X
            gen_YX: the generator for class Y to X; takes images and returns the images
                transformed to class X
            identity_criterion: the identity loss function; takes the real images from X and
                            those images put through a Y->X generator and returns the identity
                            loss (which you aim to minimize)
        '''
        #### START CODE HERE ####
        identity_X = gen_YX(real_X)
        identity_loss = identity_criterion(identity_X, real_X)
        #### END CODE HERE ####
        return identity_loss, identity_X

    def get_cycle_consistency_loss(self, real_X, fake_Y, gen_YX, cycle_criterion):
        '''
        Return the cycle consistency loss of the generator given inputs
        (and the generated images for testing purposes).
        Parameters:
            real_X: the real images from pile X
            fake_Y: the generated images of class Y
            gen_YX: the generator for class Y to X; takes images and returns the images
                transformed to class X
            cycle_criterion: the cycle consistency loss function; takes the real images from X and
                            those images put through a X->Y generator and then Y->X generator
                            and returns the cycle consistency loss (which you aim to minimize)
        '''
        #### START CODE HERE ####
        cycle_X = gen_YX(fake_Y)
        cycle_loss = cycle_criterion(cycle_X, real_X)
        #### END CODE HERE ####
        return cycle_loss, cycle_X

    def get_gen_loss(self, real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, identity_criterion,
                     cycle_criterion, lambda_identity=0.1, lambda_cycle=10):
        '''
        Return the loss of the generator given inputs.
        Parameters:
            real_A: the real images from pile A
            real_B: the real images from pile B
            gen_AB: the generator for class A to B; takes images and returns the images
                transformed to class B
            gen_BA: the generator for class B to A; takes images and returns the images
                transformed to class A
            disc_A: the discriminator for class A; takes images and returns real/fake class A
                prediction matrices
            disc_B: the discriminator for class B; takes images and returns real/fake class B
                prediction matrices
            adv_criterion: the adversarial loss function; takes the discriminator
                predictions and the true labels and returns a adversarial
                loss (which you aim to minimize)
            identity_criterion: the reconstruction loss function used for identity loss
                and cycle consistency loss; takes two sets of images and returns
                their pixel differences (which you aim to minimize)
            cycle_criterion: the cycle consistency loss function; takes the real images from X and
                those images put through a X->Y generator and then Y->X generator
                and returns the cycle consistency loss (which you aim to minimize).
                Note that in practice, cycle_criterion == identity_criterion == L1 loss
            lambda_identity: the weight of the identity loss
            lambda_cycle: the weight of the cycle-consistency loss
        '''
        # Hint 1: Make sure you include both directions - you can think of the generators as collaborating
        # Hint 2: Don't forget to use the lambdas for the identity loss and cycle loss!
        #### START CODE HERE ####
        # Adversarial Loss -- get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion)
        adv_loss_BA, fake_A = self.get_gen_adversarial_loss(real_B, disc_A, gen_BA, adv_criterion)
        adv_loss_AB, fake_B = self.get_gen_adversarial_loss(real_A, disc_B, gen_AB, adv_criterion)
        gen_adversarial_loss = adv_loss_BA + adv_loss_AB

        # Identity Loss -- get_identity_loss(real_X, gen_YX, identity_criterion)
        identity_loss_A, identity_A = self.get_identity_loss(real_A, gen_BA, identity_criterion)
        identity_loss_B, identity_B = self.get_identity_loss(real_B, gen_AB, identity_criterion)
        gen_identity_loss = identity_loss_A + identity_loss_B

        # Cycle-consistency Loss -- get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion)
        cycle_loss_BA, cycle_A = self.get_cycle_consistency_loss(real_A, fake_B, gen_BA, cycle_criterion)
        cycle_loss_AB, cycle_B = self.get_cycle_consistency_loss(real_B, fake_A, gen_AB, cycle_criterion)
        gen_cycle_loss = cycle_loss_BA + cycle_loss_AB

        # Total loss
        gen_loss = lambda_identity * gen_identity_loss + lambda_cycle * gen_cycle_loss + gen_adversarial_loss
        #### END CODE HERE ####
        return gen_loss, fake_A, fake_B

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def load_checkpoint(self):
        os.chdir(self.load_dir)

        pre_dict = torch.load(self.gen_disc_load)

        self.gen_AB.load_state_dict(pre_dict['gen_AB'])
        self.gen_BA.load_state_dict(pre_dict['gen_BA'])
        self.gen_opt.load_state_dict(pre_dict['gen_opt'])

        self.disc_A.load_state_dict(pre_dict['disc_A'])
        self.disc_A_opt.load_state_dict(pre_dict['disc_A_opt'])
        self.disc_B.load_state_dict(pre_dict['disc_B'])
        self.disc_B_opt.load_state_dict(pre_dict['disc_B_opt'])

        self.act = pre_dict['epoch']

        self.dis_loss = pre_dict['disc_loss']

        self.gen_loss = pre_dict['gen_loss']

        os.chdir('..')

    def saveCheckpoint(self, epoch):
        os.chdir(self.log_dir)

        torch.save({
            'gen_AB': self.gen_AB.state_dict(),
            'gen_BA': self.gen_BA.state_dict(),
            'gen_opt': self.gen_opt.state_dict(),
            'disc_A': self.disc_A.state_dict(),
            'disc_A_opt': self.disc_A_opt.state_dict(),
            'disc_B': self.disc_B.state_dict(),
            'disc_B_opt': self.disc_B_opt.state_dict(),
            'epoch': epoch,
            'gen_loss': self.gen_loss,
            'disc_loss': self.dis_loss
        }, 'cycleGAN_' + str(epoch) + '.pth')

        os.chdir('..')

    def check_per_batch(self, real, generated, it):
        return

    def get_InputVector_paraEtiquetar(self, etiquetas, numClases):
        return F.one_hot(etiquetas, numClases)

    def combinarVectores(self, x, y):
        return torch.cat((x.float(), y.float()), 1)

    def train_gen(self, real_A, real_B):
        self.enable_training(self.gen_BA, True)
        self.enable_training(self.gen_AB, True)
        self.enable_training(self.disc_A, False)
        self.enable_training(self.disc_B, False)

        ### Update generator ###
        self.gen_opt.zero_grad()
        gen_loss, fake_A, fake_B = self.get_gen_loss(
            real_A, real_B, self.gen_AB, self.gen_BA, self.disc_A, self.disc_B, self.adv_criterion,
            self.recon_criterion, self.recon_criterion
        )
        gen_loss.backward()  # Update gradients
        self.gen_opt.step()  # Update optimizer

        return (gen_loss, fake_A, fake_B)

    def train_disc(self, real_A, real_B):
        self.enable_training(self.gen_BA, False)
        self.enable_training(self.gen_AB, False)
        self.enable_training(self.disc_A, True)
        self.enable_training(self.disc_B, True)

        ### Update discriminator A ###
        self.disc_A_opt.zero_grad()  # Zero out the gradient before backpropagation
        with torch.no_grad():
            fake_A = self.gen_BA(real_B)
        disc_A_loss = self.get_disc_loss(real_A, fake_A, self.disc_A, self.adv_criterion)
        disc_A_loss.backward(retain_graph=True)  # Update gradients
        self.disc_A_opt.step()  # Update optimizer

        ### Update discriminator B ###
        self.disc_B_opt.zero_grad()  # Zero out the gradient before backpropagation
        with torch.no_grad():
            fake_B = self.gen_AB(real_A)
        disc_B_loss = self.get_disc_loss(real_B, fake_B, self.disc_B, self.adv_criterion)
        disc_B_loss.backward(retain_graph=True)  # Update gradients
        self.disc_B_opt.step()  # Update optimizer

        return disc_A_loss, disc_B_loss

    def train_for_epochs(self, n_epochs):
        self.initial__time = time.time()
        self.mean_generator_loss = 0
        self.mean_discriminator_loss = 0
        for e in range(0, n_epochs):
            self.epoch(e + self.act)
            self.act += 1

    def epoch(self, ep):
        it = 0
        for (real_A, real_B) in tqdm(zip(self.dataloader1, self.dataloader2)):
            real_A = nn.functional.interpolate(real_A, size=self.target_shape)
            real_B = nn.functional.interpolate(real_B, size=self.target_shape)
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)

            generator_loss, fake_A, fake_B = self.train_gen(real_A, real_B)
            disc_A_loss, disc_B_loss = self.train_disc(real_A, real_B)

            # Keep track of the average discriminator loss
            self.mean_discriminator_loss += disc_A_loss.item() / self.log_step
            # Keep track of the average generator loss
            self.mean_generator_loss += generator_loss.item() / self.log_step

            # Keep track of the generator losses
            self.gen_loss.append(generator_loss.item())
            # Keep track of the average discriminator loss
            self.dis_loss.append(disc_A_loss.item())

            self.iter = self.iter + 1
            it = it + 1

            ### Visualization code ###
            if self.iter % self.log_step == 0 and self.iter > 0:
                self.plot_losses()
                self.save_results(torch.cat([real_A, real_B]), torch.cat([fake_B, fake_A]))

                if self.time_steps:
                    self.plot_step_time()

            if self.checksave and self.iter % self.save_step == 0:
                self.saveCheckpoint(ep)

        if self.time_epochs:
            self.plot_epoch_time()

        self.check_per_batch(torch.cat([real_A, real_B]), torch.cat([fake_B, fake_A]), it)

    def save_results(self, real, generated):
        ImageFunctions.tensor_as_image(generated, self.iter, "fake", self.log_dir, save=True, show=False)
        ImageFunctions.tensor_as_image(real, self.iter, "real", self.log_dir, save=True, show=False)

    def enable_training(self, model, flag):
        for p in model.parameters():
            p.requires_grad = flag

    def check_training_params(self, model, flag):
        for p in model.parameters():
            assert (p.requires_grad == flag)

    def plot_epoch_time(self):
        epoch_time = (time.time() - self.initial__time)
        self.epoch_times.append(epoch_time)
        self.num_epochs.append(self.act)

        title = str(self.epoch_times[-1]) + " seconds taken"
        print(title)

        plt.plot(self.num_epochs, self.epoch_times, label="Epoch Time")

        plt.title(title)
        plt.legend()

        os.chdir(self.log_dir)
        plt.savefig(datetime.now().strftime("%d-%m") + " iter " + str(self.act) + " epoch_times" + '.pdf')
        os.chdir('..')

        plt.clf()

    def plot_step_time(self):

        step_time = (time.time() - self.initial__time)
        self.step_times.append(step_time)
        self.num_steps.append(self.iter)

        title = str(self.step_times[-1]) + " seconds to take " + str(self.iter) + " steps"
        print(title)

        # Time visualization

        plt.plot(self.num_steps, self.step_times, label="Step Time")

        plt.title(title)
        plt.legend()

        os.chdir(self.log_dir)
        plt.savefig(datetime.now().strftime("%d-%m") + " iter " + str(self.iter) + " step_times" + '.pdf')
        os.chdir('..')

        plt.clf()

    def plot_losses(self):

        self.mean_generator_loss = 0
        self.mean_discriminator_loss = 0
        step_bins = 20
        x_axis = sorted([i * step_bins for i in range(len(self.gen_loss) // step_bins)] * step_bins)
        num_examples = (len(self.gen_loss) // step_bins) * step_bins
        plt.plot(
            range(num_examples // step_bins),
            torch.Tensor(self.gen_loss[:num_examples]).view(-1, step_bins).mean(1),
            label="Generator Loss"
        )
        plt.plot(
            range(num_examples // step_bins),
            torch.Tensor(self.dis_loss[:num_examples]).view(-1, step_bins).mean(1),
            label="Discriminator Loss"
        )
        plt.legend()

        os.chdir(self.log_dir)
        plt.savefig(datetime.now().strftime("%d-%m") + " iter " + str(self.iter) + '.pdf')
        os.chdir('..')
        plt.clf()

    def generate_samples(self, n_samples=1):
        i = 0
        for (real_A, real_B) in tqdm(zip(self.dataloader1, self.dataloader2)):
            i += 1
            real_A = nn.functional.interpolate(real_A, size=self.target_shape)
            real_B = nn.functional.interpolate(real_B, size=self.target_shape)
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)
            gen_loss, fake_A, fake_B = self.get_gen_loss(
                real_A, real_B, self.gen_AB, self.gen_BA, self.disc_A, self.disc_B, self.adv_criterion,
                self.recon_criterion, self.recon_criterion
            )
            self.save_results(torch.cat([real_A, real_B]), torch.cat([fake_B, fake_A]))

            if i > n_samples:
                break