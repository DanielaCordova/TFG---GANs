
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss



class GANLoss:
    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, height, alpha):

        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, height, alpha):

        raise NotImplementedError("gen_loss method has not been implemented")


class ConditionalGANLoss:

    def __init__(self, dis):
        self.criterion = BCEWithLogitsLoss()
        self.dis = dis

    def dis_loss(self, realImgs, fakeImgs, labels, height, alpha):

        device = fakeImgs.device

        # Predictions by Discriminator
        real_preds = self.dis(realImgs, height, alpha, labels_in=labels)
        fake_preds = self.dis(fakeImgs, height, alpha, labels_in=labels)

        # Applies BCE in real
        real_loss = self.criterion(torch.squeeze(real_preds), torch.ones(realImgs.shape[0]).to(device))

        # Applies BCE in Fake
        fake_loss = self.criterion(torch.squeeze(fake_preds), torch.zeros(fakeImgs.shape[0]).to(device))

        #La media
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fakeImgs, labels, height, alpha):
        preds = self.dis(fakeImgs, height, alpha, labels_in=labels)
        return self.criterion(torch.squeeze(preds), torch.ones(fakeImgs.shape[0]).to(fakeImgs.device)) ##Only BCE: predict Dis vs FakeImg


class LogisticGAN(GANLoss):
    def __init__(self, dis):
        super().__init__(dis)

    # gradient penalty
    def R1Penalty(self, real_img, height, alpha):
        real_img = torch.autograd.Variable(real_img, requires_grad=True)
        ##Penalty only in Discriminator
        real_logit = self.dis(real_img, height, alpha)

        real_grads = torch.autograd.grad(outputs=real_logit, inputs=real_img,
                                         grad_outputs=torch.ones(real_logit.size()).to(real_img.device),
                                         create_graph=True, retain_graph=True)[0].view(real_img.size(0), -1)

        r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
        return r1_penalty

    def dis_loss(self, real_samps, fake_samps, height, alpha, r1_gamma=10.0):

        real_preds = self.dis(real_samps, height, alpha)
        fake_preds = self.dis(fake_samps, height, alpha)

        loss = torch.mean(nn.Softplus()(fake_preds)) + torch.mean(nn.Softplus()(-real_preds))

        if r1_gamma != 0.0: #Applies R1Penalty
            r1_penalty = self.R1Penalty(real_samps.detach(), height, alpha) * (r1_gamma * 0.5)
            loss += r1_penalty

        return loss

    def gen_loss(self, _, fake_samps, height, alpha):
        fake_preds = self.dis(fake_samps, height, alpha)

        return torch.mean(nn.Softplus()(-fake_preds))
