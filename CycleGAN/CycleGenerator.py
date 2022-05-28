import sys, os

curentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(curentdir)
sys.path.append(parentdir)

import torch
import torch.nn as nn
import Blocks as bk

class Generator(nn.Module):
    '''
    Generator Class
    A series of 2 contracting blocks, 9 residual blocks, and 2 expanding blocks to 
    transform an input image into an image from the other class, with an upfeature
    layer at the start and a downfeature layer at the end.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels, hidden_channels=32):
        super(Generator, self).__init__()
        self.upfeature = bk.FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = bk.ContractingBlock(hidden_channels)
        self.contract2 = bk.ContractingBlock(hidden_channels * 2)
        res_mult = 4
        self.res0 = bk.ResidualBlock(hidden_channels * res_mult)
        self.res1 = bk.ResidualBlock(hidden_channels * res_mult)
        self.res2 = bk.ResidualBlock(hidden_channels * res_mult)
        self.res3 = bk.ResidualBlock(hidden_channels * res_mult)
        self.res4 = bk.ResidualBlock(hidden_channels * res_mult)
        self.res5 = bk.ResidualBlock(hidden_channels * res_mult)
        self.res6 = bk.ResidualBlock(hidden_channels * res_mult)
        self.res7 = bk.ResidualBlock(hidden_channels * res_mult)
        self.res8 = bk.ResidualBlock(hidden_channels * res_mult)
        
        self.expand2 = bk.ExpandingBlock(hidden_channels * 4)
        self.expand3 = bk.ExpandingBlock(hidden_channels * 2)
        self.downfeature = bk.FeatureMapBlock(hidden_channels, output_channels)
        self.tanh = nn.Tanh()

    def forward(self, x):
        '''
        Function for completing a forward pass of Generator: 
        Given an image tensor, passes it through the U-Net with residual blocks
        and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.res0(x2)
        x4 = self.res1(x3)
        x5 = self.res2(x4)
        x6 = self.res3(x5)
        x7 = self.res4(x6)
        x8 = self.res5(x7)
        x9 = self.res6(x8)
        x10 = self.res7(x9)
        x11 = self.res8(x10)
        x12 = self.expand2(x11)
        x13 = self.expand3(x12)
        xn = self.downfeature(x13)
        return self.tanh(xn)