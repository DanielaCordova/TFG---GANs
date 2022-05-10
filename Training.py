
from abc import abstractclassmethod
from venv import create
import tqdm
from torch.utils.data import DataLoader

import Constants
import StyleComponents
import torch
import ImageFunctions
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

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
        plt.savefig(datetime.now().strftime("%d-%m")+" iter " + str(self.act) + " epoch_times" + '.svg')
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
        plt.savefig(datetime.now().strftime("%d-%m")+" iter " + str(self.iter) + " step_times" + '.svg')
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


        self.gen.load_state_dict(c_g['model_state_dict'])
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
        plt.savefig(datetime.now().strftime("%d-%m")+" iter " + str(self.iter) + '.svg')
        os.chdir('..')

        plt.clf()
        
        

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


        self.gen.load_state_dict(c_g['model_state_dict'])
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
        plt.savefig(datetime.now().strftime("%d-%m")+" iter " + str(self.iter) + '.svg')
        os.chdir('..')

        plt.clf()


class Cycle_Trainer(GAN_Trainer):
    def __init__(self, dataloader, generator, discriminator, criterion, log_step, log_dir, num_classes, device='cuda',
                 verb=False, checksave=False, save_step=None, load=False, load_dir=None, gen_load=None, disc_load=None,
                 time_steps=False, time_epochs=False):

        super().__init__(dataloader, generator, discriminator, criterion, log_step, log_dir, checksave, save_step, load,
                         load_dir, gen_load, disc_load, time_steps, time_epochs, device=device)
        self.verb = verb
        self.num_classes = num_classes

        if verb:
            self.dis_fake_loss_plot = []
            self.dis_real_loss_plot = []
            self.dis_real_loss = []
            self.dis_fake_loss = []

        if load:
            self.load_checkpoint()

    def load_checkpoint(self):
        os.chdir(self.load_dir)

        c_g = torch.load(self.gen_load)
        c_d = torch.load(self.disc_load)

        self.act = c_d['epoch']

        self.discriminator.load_state_dict(c_d['model_state_dict'])
        self.disc_opt.load_state_dict(c_d['optimizer_state_dict'])
        self.dis_loss.append(c_d['loss'])

        self.gen.load_state_dict(c_g['model_state_dict'])
        self.gen_opt.load_state_dict(c_g['optimizer_state_dict'])
        self.gen_loss.append(c_g['loss'])

        os.chdir('..')

    def saveCheckpoint(self, epoch):
        os.chdir(self.log_dir)

        g_s = 'gen_' + str(epoch) + '.tar'
        d_s = 'disc_' + str(epoch) + '.tar'

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.gen_opt.state_dict(),
            'loss': self.gen_loss[-1]
        }, g_s)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.disc_opt.state_dict(),
            'loss': self.dis_loss[-1]
        }, d_s)

        os.chdir('..')

    def preprocessRealData(self, real_data):
        real, tag = real_data

        tag = torch.nn.functional.one_hot(tag, self.num_classes)
        img_vec_tag = tag[:, :, None, None]
        img_vec_tag = img_vec_tag.repeat(1, 1, real.shape[2], real.shape[3])
        img_vec_tag = torch.cat((real.float(), img_vec_tag.float()), 1)

        return img_vec_tag, tag

    def appendDiscLoss(self, discLoss):

        if self.verb:
            total_loss, real_loss, fake_loss = discLoss
            self.dis_real_loss.append(real_loss)
            self.dis_fake_loss.append(fake_loss)
            self.dis_loss.append(total_loss)
        else:
            self.dis_loss.append(total_loss)

    def check_per_batch(self, real, generated, it):
        return

    def get_InputVector_paraEtiquetar(self, etiquetas, numClases):
        return F.one_hot(etiquetas, numClases)

    def combinarVectores(self, x, y):
        return torch.cat((x.float(), y.float()), 1)

    def train_gen(self, real):
        self.enable_training(self.generator, True)
        self.enable_training(self.discriminator, False)

        real, tag = real

        real = real.to(self.device)
        tag = tag.to(self.device)
        noise = torch.randn((Constants.BATCH_SIZE, self.generator.getNoiseDim())).to(self.device)

        noise_tag = self.combinarVectores(noise, tag)

        fake = self.generator(noise_tag)

        img_vec_tag = tag[:, :, None, None]
        img_vec_tag = img_vec_tag.repeat(1, 1, fake.shape[2], fake.shape[3])
        fake_tag = torch.cat((fake.float(), img_vec_tag.float()), 1)
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
        tag = tag.to(self.device)
        noise = torch.randn((Constants.BATCH_SIZE, self.generator.getNoiseDim())).to(self.device)

        noise_tag = self.combinarVectores(noise, tag)

        fake = self.generator(noise_tag)

        img_vec_tag = tag[:, :, None, None]
        img_vec_tag = img_vec_tag.repeat(1, 1, fake.shape[2], fake.shape[3])
        fake_tag = torch.cat((fake.float(), img_vec_tag.float()), 1)
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
            if real[0].shape[0] != Constants.BATCH_SIZE:
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
            if self.iter % self.log_step == 0 and self.iter > 0:
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

        self.dis_plot_loss.append(sum(self.dis_loss[-self.log_step:]) / self.log_step)
        self.gen_plot_loss.append(sum(self.gen_loss[-self.log_step:]) / self.log_step)

        if self.verb:
            l1 = sum(self.dis_fake_loss[-self.log_step:]) / self.log_step
            l2 = sum(self.dis_real_loss[-self.log_step:]) / self.log_step
            self.dis_fake_loss_plot.append(l1)
            self.dis_real_loss_plot.append(l2)

        title = "Gen Loss: " + str(self.gen_plot_loss[-1]) + " Disc Loss: " + str(self.dis_plot_loss[-1])
        print(title)

        self.ejeX.append(self.iter)
        arEjeX = np.array(self.ejeX)
        plt.plot(arEjeX, np.array(self.gen_plot_loss), label="Gen Loss")
        plt.plot(arEjeX, np.array(self.dis_plot_loss), label="Disc Loss")

        if self.verb:
            plt.plot(arEjeX, np.array(self.dis_fake_loss_plot), label="Disc Fake Loss")
            plt.plot(arEjeX, np.array(self.dis_real_loss_plot), label="Disc Real Loss")

        plt.title(title)
        plt.legend()

        os.chdir(self.log_dir)
        plt.savefig(datetime.now().strftime("%d-%m") + " iter " + str(self.iter) + '.svg')
        os.chdir('..')

        plt.clf()

class Style_Prog_Trainer:
    def __init__(self, dataset, generator, discriminator, criterion, dir, verb, prog, device, 
    checksave = False, load = False, load_dir = None, gen_load = None, disc_load = None, time_steps = True, time_epochs = True):
        
        self.disc = discriminator
        self.gen  = generator
        self.criterion = criterion

        self.device = device

        self.dataset = dataset

        self.resdir = dir
        self.log_step = 0

        self.iter = 0

        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=Constants.LR, betas=(0, 0.99))
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=Constants.LR,  betas=(0, 0.99))
        self.downsampler = nn.AvgPool2d(4,2,1)

        self.verb = verb
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

        if prog :
            self.increase_alfa_step = 0
            self.alfa_step = 0
            self.prog = prog
        else:
            self.prog = False
            self.alfa_step = 0
            self.increase_alfa_step = 0


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


    def train_for_epochs(self, epochs_for_depth, batch_size_for_depth):
        ini = 0
        self.num_ep = 0
        while ini < len(epochs_for_depth) and self.act_epoch > epochs_for_depth[ini]  :
            self.num_ep = self.num_ep + epochs_for_depth[ini]
            self.act_epoch = self.act_epoch - epochs_for_depth[ini]
            i = i + 1

        num_ep = self.num_ep + epochs_for_depth[ini] - self.act_epoch
        epochs_for_depth[ini] = epochs_for_depth[ini] - self.act_epoch

        for i in range(ini, len(epochs_for_depth)):
            self.initial__time = time.time()
            self.batch_size = batch_size_for_depth[i]
            self.dataloader = DataLoader(self.dataset, self.batch_size, shuffle = True)
            self.display_step = int(len(self.dataset)/self.batch_size)
            self.increase_alfa_step = 8
            self.alfa_step = (1 / (epochs_for_depth[i] * len(self.dataloader))) * self.increase_alfa_step
            self.log_step = int(len(self.dataset) / self.batch_size)
            self.display_step = int(len(self.dataset)/ self.batch_size)
            self.save_step = int(len(self.dataset)/ self.batch_size)

            print(" ## Training data for size {} ".format(self.disc.getinSize()))
            print("      Epochs for depth {} = {}".format(i, epochs_for_depth[i]))
            print("      Batch size for depth {} = {}".format(i, batch_size_for_depth[i]))
            print("      Increase_alfa_step for depth {} = {}".format(i, self.increase_alfa_step))
            print("      Alfa step for depth {} = {}".format(i, self.alfa_step))

            for _ in range(0, epochs_for_depth[i]):
                self.epoch(self.num_ep)
                self.num_ep = self.num_ep + 1
            
            self.gen.increaseDepth()
            self.disc.increaseDepth()

            print("Tamanio = " + str(self.disc.getinSize()))


    def plot_losses(self):

        self.disc_loss_plot.append(sum(self.disc_loss[-self.log_step:])/self.log_step)
        self.gen_loss_plot.append(sum(self.gen_loss[-self.log_step:])/self.log_step)

        title = "Gen Loss: " + str(self.gen_loss_plot[-1]) + " Disc Loss: " + str(self.disc_loss_plot[-1])
        print(title)
        
        self.ejeX.append(self.iter)
        arEjeX = np.array(self.ejeX)
        plt.plot(arEjeX, np.array(self.gen_loss_plot), label = "Gen Loss")
        plt.plot(arEjeX, np.array(self.disc_loss_plot), label = "Disc Loss")

        plt.title(title)
        plt.legend()

        os.chdir(self.resdir)
        plt.savefig(datetime.now().strftime("%d-%m")+" iter " + str(self.iter) + '.svg')
        os.chdir('..')

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
        'alfa' : self.gen.getAlfa(),
        'depth': self.gen.getDepth(),
        'model_state_dict' : self.gen.state_dict(), 
        'optimizer_state_dict' : self.gen_opt.state_dict(),
        'loss' : self.gen_loss[-1]
        }, g_s)

        torch.save({
            'epoch' : epoch,
            'alfa' : self.disc.getAlfa(),
            'depth': self.disc.getDepth(),
            'model_state_dict' : self.disc.state_dict(),
            'optimizer_state_dict' : self.disc_opt.state_dict(),
            'loss' : self.disc_loss[-1]
        }, d_s)

        os.chdir('..')
        
    def plot_epoch_time(self):
        epoch_time = (time.time() - self.initial__time)
        self.epoch_times.append(epoch_time)
        self.num_epochs.append(self.num_ep)

        title = str(self.epoch_times[-1]) + " seconds taken"
        print(title)

        plt.plot(self.num_epochs,self.epoch_times, label="Epoch Time")
        
        plt.title(title)
        plt.legend()

        os.chdir(self.resdir)
        plt.savefig(datetime.now().strftime("%d-%m")+" iter " + str(self.num_ep) + "epoch_times" + '.svg')
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

        os.chdir(self.resdir)
        plt.savefig(datetime.now().strftime("%d-%m")+" iter " + str(self.iter) + "step_times" + '.svg')
        os.chdir('..')

        plt.clf()

class HingeRelativisticLoss():
    
    def dis_loss(self, real_sample, fake_sample, real_pred, fake_pred):

        real_fake_d = real_pred - torch.mean(fake_pred)
        fake_real_d = fake_pred - torch.mean(real_pred)

        t1 = torch.mean(nn.ReLU()(1-real_fake_d))
        t2 = torch.mean(nn.ReLU()(1+fake_real_d))

        return (t1 + t2)

    def gen_loss(self, real_sample, fake_sample, real_pred, fake_pred):

        real_fake_d = real_pred - torch.mean(fake_pred)
        fake_real_d = fake_pred - torch.mean(real_pred)

        t1 = torch.mean(nn.ReLU()(1+real_fake_d))
        t2 = torch.mean(nn.ReLU()(1-fake_real_d))

        return (t1 + t2)

class LogisticLoss():

    def __init__(self, disc, gamma=10.0):
        self.gamma = gamma
        self.disc  = disc

    def dis_loss(self, real_sample, fake_sample, real_pred, fake_pred):

        real_sample = torch.autograd.Variable(real_sample, requires_grad=True)
        real_log = self.disc(real_sample)

        loss = torch.mean(nn.Softplus()(fake_pred)) + torch.mean(nn.Softplus()(-real_pred))

        r_gradients = torch.autograd.grad(outputs=real_log, inputs=real_sample, 
        grad_outputs=torch.ones(real_log.size()).to(real_log.device), 
        create_graph=True, retain_graph=True)[0].view(real_sample.size(0), -1)

        r_gradients = r_gradients.view(real_sample.size(0), -1)

        penalty = torch.sum(torch.mul(r_gradients, r_gradients)) * (self.gamma * 0.5)

        loss = loss + penalty

        return loss

    def gen_loss(self, real_sample, fake_sample, real_pred, fake_pred):
        return torch.mean(nn.Softplus()(-fake_pred))
