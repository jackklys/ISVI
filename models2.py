from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from models1 import IWAE, make_net, fc_block

class IWAE2(IWAE):
    def __init__(self, opts):
        super(IWAE2, self).__init__(opts)
        self.encoder = Encoder(opts['x_dim'], opts['h_dim'], opts['z_dim'], opts['num_samples'])
        self.decoder = Decoder(opts['x_dim'], opts['h_dim'], opts['z_dim'], opts['num_samples'])

class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, num_samples):
        super(Encoder, self).__init__()

        self.net = make_net(x_dim, h_dim, z_dim)
        self.fc1 = nn.Linear(h_dim, z_dim * num_samples)
        self.fc2 = nn.Linear(h_dim, z_dim * num_samples)

    def forward(self, x):
        z = self.net.forward(x)
        return self.fc1(z), torch.exp(self.fc2(z))

class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, num_samples):
        super(Decoder, self).__init__()

        self.net = make_net(z_dim, h_dim, x_dim)
        self.fc = nn.Linear(h_dim, x_dim)

    def forward(self, x):
        z = self.fc(self.net.forward(x))
        return torch.sigmoid(z)