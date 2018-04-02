from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import math

class VAECOV2(nn.Module):
    def __init__(self, opts):
        super(VAECOV2, self).__init__()
        self.encoder = Encoder(opts['x_dim'], opts['h_dim'], opts['z_dim'], opts['num_samples'])
        self.decoder = Decoder(opts['x_dim'], opts['h_dim'], opts['z_dim'], opts['num_samples'])
        self.x_dim = opts['x_dim']
        self.z_dim = opts['z_dim']
        self.num_samples = opts['num_samples']

    def encode(self, x):
        mu, logvar = self.encoder.forward(x)
        return mu, logvar

    def sample(self, mu, logvar):
        def erfinv(x):
            a_for_erf = 8.0/(3.0*np.pi)*(np.pi-3.0)/(4.0-np.pi)
            b = -2/(np.pi*a_for_erf)-torch.log(1-x*x)/2
            return torch.sign(x)*torch.sqrt(b+torch.sqrt(b*b-torch.log(1-x*x)/a_for_erf))

        if self.training:
            std = logvar.unsqueeze(1).mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).uniform_(0,1)).cuda()
            eps1 = 1 - eps
            eps2 = torch.cat([eps[:,:,int(self.z_dim/2):], eps[:,:,:int(self.z_dim/2)]], 2)
            qnt = math.sqrt(2) * erfinv(2 * torch.cat([eps, eps1, eps2], 1) - 1)
            rnd = mu.unsqueeze(1) + std * qnt
            return rnd.view(-1, rnd.size(-1))
        else:
            return mu

    def decode(self, z):
        return self.decoder.forward(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.x_dim))
        z = self.sample(mu, logvar)
        return self.decode(z), mu, logvar, z

    def loss(self, data):
        recon_x, mu, logvar, samples= self.forward(data)

        x = data.view(data.size(0), 1, self.x_dim)
        recon_x = recon_x.view(x.size(0), -1, self.x_dim)
        samples = samples.view(x.size(0), -1, self.z_dim)
        mu = mu.unsqueeze(1)
        logvar = logvar.unsqueeze(1)

        logpxIh = - ((x - recon_x) ** 2).sum(-1) / 2 
        logph = - ((samples ** 2)  / 2 + 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logpxIh + logph
        logq = - (((samples - mu) ** 2) / (2 * torch.exp(logvar)) + 0.5 * logvar).sum(-1)

        def log_sum_exp(v):
            m = torch.max(v, 1)[0].unsqueeze(1)
            l = torch.sum(torch.exp(v - m), 1)
            return torch.log(l) + m

        r = log_sum_exp(logp - logq) - torch.log(Variable(torch.Tensor([self.num_samples])).cuda())
        r = torch.mean(r)

        return -r, recon_x

class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, num_samples):
        super(Encoder, self).__init__()

        def fc_block(in_dim, out_dim, dropout=False):
            modules = [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            ]

            if dropout:
                modules.append(torch.nn.Dropout())

            modules += [
                nn.ReLU()
            ]

            return nn.Sequential(*modules)

        self.net = nn.Sequential(
            fc_block(x_dim, h_dim),
            fc_block(h_dim, h_dim),
            fc_block(h_dim, z_dim),
        )

        self.fc1 = nn.Linear(z_dim, z_dim)
        self.fc2 = nn.Linear(z_dim, 1)

    def forward(self, x):
        z = self.net.forward(x)
        return self.fc1(z), self.fc2(z).expand(z.size(0), z.size(1))

class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, num_samples):
        super(Decoder, self).__init__()

        def fc_block(in_dim, out_dim, dropout=False):
            modules = [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            ]

            if dropout:
                modules.append(torch.nn.Dropout())

            modules += [
                nn.ReLU()
            ]

            return nn.Sequential(*modules)

        self.net = nn.Sequential(
            fc_block(z_dim, h_dim),
            fc_block(h_dim, h_dim),
            fc_block(h_dim, x_dim),
        )

    def forward(self, x):
        return self.net.forward(x)