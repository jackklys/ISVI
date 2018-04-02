from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from models1 import IWAE, COV, fc_block, make_net, kronecker_product

class COVK(COV):
    def __init__(self, opts):
        super(COVK, self).__init__(opts)
        self.encoder = Encoder(opts['x_dim'], opts['h_dim'], opts['z_dim'], opts['num_samples'])
        self.decoder = Decoder(opts['x_dim'], opts['h_dim'], opts['z_dim'], opts['num_samples'])
        self.b = opts['num_samples']

    def encode(self, x):
        mu, std, B = self.encoder.forward(x.view(-1, self.x_dim))

        # creating A = diag(std)
        A = torch.cat([std.unsqueeze(2),Variable(torch.zeros(std.size(0), std.size(1), std.size(1))).cuda()], 2)
        A = A.view(-1, std.size(1) + 1, std.size(1))
        A = A[:,:-1,:]

        # expanding B if necessary
        B = B.view(-1, self.b, self.b)
        n = self.num_samples /self.b
        B = kronecker_product(Variable(torch.eye(n).expand(B.size(0), n, n)).cuda(), B)

        # normalizing rows of B
        B = torch.sqrt(1. / torch.sum(B*B, 2, keepdim=True)) * B

        L = kronecker_product(B, A)

        mu = mu.view(mu.size(0), -1, self.z_dim)
        std = std.view(mu.size(0), -1, self.z_dim)
        mu = mu.expand(-1, self.num_samples, self.z_dim)
        std = std.expand(-1, self.num_samples, self.z_dim)

        return mu, std, L

class COVK2(COVK):
    def __init__(self, opts):
        super(COVK2, self).__init__(opts)
        self.b = 2
        self.encoder = Encoder(opts['x_dim'], opts['h_dim'], opts['z_dim'], self.b)
        self.decoder = Decoder(opts['x_dim'], opts['h_dim'], opts['z_dim'], self.b)

class COVK3(COVK):
    def __init__(self, opts):
        super(COVK3, self).__init__(opts)
        self.b = 3
        self.encoder = Encoder(opts['x_dim'], opts['h_dim'], opts['z_dim'], self.b)
        self.decoder = Decoder(opts['x_dim'], opts['h_dim'], opts['z_dim'], self.b)

class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, b):
        super(Encoder, self).__init__()

        self.net = make_net(x_dim, h_dim, z_dim)
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, (b**2))

    def forward(self, x):
        z = self.net.forward(x)
        return self.fc1(z), torch.exp(self.fc2(z)), self.fc3(z)

class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, num_samples):
        super(Decoder, self).__init__()

        self.net = make_net(z_dim, h_dim, x_dim)
        self.fc = nn.Linear(h_dim, x_dim)

    def forward(self, x):
        z = self.fc(self.net.forward(x))
        return torch.sigmoid(z)