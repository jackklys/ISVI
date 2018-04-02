from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from models1 import fc_block, make_net

class VAECOV1(nn.Module):
    def __init__(self, opts):
        super(VAECOV1, self).__init__()
        self.encoder = Encoder(opts['x_dim'], opts['h_dim'], opts['z_dim'], opts['num_samples'])
        self.decoder = Decoder(opts['x_dim'], opts['h_dim'], opts['z_dim'], opts['num_samples'])
        self.x_dim = opts['x_dim']
        self.z_dim = opts['z_dim']
        self.num_samples = opts['num_samples']

        A1 = torch.eye(self.z_dim).repeat(self.num_samples, self.num_samples)
        # A1 = torch.eye(self.z_dim * self.num_samples)
        A1 = A1.expand(opts['batch_size'], A1.size(0), A1.size(1))
        self.n1 = A1.nonzero().type(torch.cuda.LongTensor)


    def encode(self, x):
        z, std1 = self.encoder.forward(x)
        n1 = self.n1[:z.size(0) * self.z_dim * (self.num_samples ** 2)]
        A1 = Variable(torch.zeros(z.size(0), self.z_dim * self.num_samples, self.z_dim * self.num_samples)).cuda()
        A1[n1[:,0], n1[:,1], n1[:,2]] = std1.view(-1)
        return z, A1

    def sample(self, mu, A):
        if self.training:
            eps = Variable(torch.normal(torch.zeros(A.size(0), self.num_samples * self.z_dim, 1), std=1.0)).cuda()
            std = torch.matmul(A, eps)
            # std = torch.matmul(torch.exp(0.5 * A), Variable(torch.randn(A.size(0), self.num_samples * self.z_dim, 1)).cuda())
            # std = std.view(mu.size(0), self.num_samples, self.z_dim)
            rnd = mu.unsqueeze(2) + std
            return rnd.view(-1, self.z_dim)
        else:
            return mu.view(-1, self.z_dim)

    def decode(self, z):
        return self.decoder.forward(z)

    def forward(self, x):
        mu, A = self.encode(x.view(-1, self.x_dim))
        z = self.sample(mu, A)
        cov = torch.matmul(A, torch.transpose(A, 1, 2))
        # torch.set_printoptions(threshold=50000)
        # print(cov[0])
        mask = Variable(torch.eye(cov.size(1), cov.size(2)).type(torch.ByteTensor)).cuda()
        logvar = torch.log(torch.masked_select(cov, mask).view(-1, self.z_dim))
        # logvar = torch.masked_select(A, mask).view(-1, self.z_dim)
        return self.decode(z), mu, logvar, z

    def loss(self, data):
        recon_x, mu, logvar, samples= self.forward(data)

        x = data.view(data.size(0), 1, self.x_dim)
        recon_x = recon_x.view(x.size(0), -1, self.x_dim)
        samples = samples.view(x.size(0), -1, self.z_dim)

        mu = mu.view(x.size(0), -1, self.z_dim)
        logvar = logvar.view(x.size(0), -1, self.z_dim)

        a = x * torch.log(recon_x + 1e-18) + (1 - x) * torch.log(1 - recon_x + 1e-18)
        logpxIh = a.sum(-1)
        logph = - ((samples ** 2)  / 2 + 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logpxIh + logph
        logq = - (((samples - mu) ** 2) / (2 * torch.exp(logvar)) + 0.5 * logvar + 0.5 * np.log(2 * np.pi)).sum(-1)

        def log_mean_exp(v):
            m = torch.max(v, 1)[0].unsqueeze(1)
            return torch.log(torch.mean(torch.exp(v - m), 1)) + m

        r = log_mean_exp(logp - logq)
        r = torch.mean(r)

        return -r, recon_x

class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, num_samples):
        super(Encoder, self).__init__()

        self.net = make_net(x_dim, h_dim, z_dim)
        self.fc1 = nn.Linear(h_dim, z_dim * num_samples)
        self.fc2 = nn.Linear(h_dim, z_dim * (num_samples**2))

    def forward(self, x):
        z = self.net.forward(x)
        return self.fc1(z), self.fc2(z)

class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, num_samples):
        super(Decoder, self).__init__()

        self.net = make_net(z_dim, h_dim, x_dim)
        self.fc = nn.Linear(h_dim, x_dim)

    def forward(self, x):
        z = self.fc(self.net.forward(x))
        return torch.sigmoid(z)