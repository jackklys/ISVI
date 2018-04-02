import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

class IWAE(nn.Module):
    def __init__(self, opts):
        super(IWAE, self).__init__()
        self.encoder = Encoder(opts['x_dim'], opts['h_dim'], opts['z_dim'])
        self.decoder = Decoder(opts['x_dim'], opts['h_dim'], opts['z_dim'])
        self.x_dim = opts['x_dim']
        self.z_dim = opts['z_dim']
        self.num_samples = opts['num_samples']

    def encode(self, x):
        mu, std = self.encoder.forward(x.view(-1, self.x_dim))
        mu = mu.view(mu.size(0), -1, self.z_dim)
        std = std.view(mu.size(0), -1, self.z_dim)
        mu = mu.expand(-1, self.num_samples, self.z_dim)
        std = std.expand(-1, self.num_samples, self.z_dim)
        return mu, std

    def sample(self, mu, std):
        if self.training:
          eps = Variable(torch.normal(torch.zeros(std.size(0), self.num_samples, self.z_dim), std=1.0)).cuda()
          return  mu + eps * std
        else:
          return mu

    def decode(self, z):
        z.contiguous()
        x = self.decoder.forward(z.view(-1, self.z_dim))
        return x.view(-1, self.num_samples, self.x_dim)

    def forward(self, x):
        mu, std = self.encode(x)
        samples = self.sample(mu, std)
        logvar = torch.log(std * std + 1e-18)
        recon = self.decode(samples)
        return recon, mu, logvar, samples

    def loss(self, data):
        recon, mu, logvar, samples= self.forward(data)
        x = data.view(-1, 1, self.x_dim)

        logpxIh = (x * torch.log(recon + 1e-18) + (1 - x) * torch.log(1 - recon + 1e-18)).sum(-1)
        logph = - ((samples ** 2)  / 2 + 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logpxIh + logph
        logq = - (((samples - mu) ** 2) / (2 * torch.exp(logvar)) + 0.5 * logvar + 0.5 * np.log(2 * np.pi)).sum(-1)

        def log_mean_exp(v):
            m = torch.max(v, 1)[0]
            return torch.log(torch.mean(torch.exp(v - m.unsqueeze(1)), 1)) + m

        loss = log_mean_exp(logp - logq)
        loss = - torch.mean(loss)

        return loss, recon

class IWAEN(IWAE):
    def __init__(self, opts):
        super(IWAEN, self).__init__(opts)

    def sample(self, mu, std):
        if self.training:
          eps1 = Variable(torch.normal(torch.zeros(std.size(0), int(self.num_samples/2), self.z_dim), std=1.0)).cuda()
          eps2 = -eps1
          eps = torch.cat([eps1, eps2], 1)
          return mu + eps * std
        else:
          return mu

class COV(IWAE):
    def __init__(self, opts):
        super(COV, self).__init__(opts)
        self.b = self.num_samples * self.z_dim
        self.B = Variable(torch.eye(self.b)).cuda()

    def encode(self, x):
        mu, std = self.encoder.forward(x.view(-1, self.x_dim))

        #make B
        B = self.B.expand(mu.size(0), self.b, self.b)
        B = torch.sqrt(1. / torch.sum(B*B, 2)).unsqueeze(2) * B
        n = self.num_samples * self.z_dim / self.b
        B = kronecker_product(Variable(torch.eye(n).expand(mu.size(0), n, n)).cuda(), B)
        B = B * std.repeat(1, self.num_samples).unsqueeze(2)

        mu = mu.view(mu.size(0), -1, self.z_dim)
        std = std.view(mu.size(0), -1, self.z_dim)
        mu = mu.expand(-1, self.num_samples, self.z_dim)
        std = std.expand(-1, self.num_samples, self.z_dim)

        return mu, std, B

    def sample(self, mu, L):
        if self.training:
            eps = Variable(torch.normal(torch.zeros(L.size(0), self.num_samples * self.z_dim, 1), std=1.0)).cuda()
            std = torch.matmul(L, eps).view(-1, self.num_samples, self.z_dim)
            return mu + std
        else:
            return mu

    def forward(self, x):
        mu, std, L = self.encode(x.view(-1, self.x_dim))
        samples = self.sample(mu, L)
        logvar = torch.log(std ** 2 + 1e-18)
        return self.decode(samples), mu, logvar, samples

class COVD(COV):
    def __init__(self, opts):
        super(COVD, self).__init__(opts)
        self.encoder = Encoder(opts['x_dim'], opts['h_dim'], opts['z_dim'])
        self.decoder = Decoder(opts['x_dim'], opts['h_dim'], opts['z_dim'])
        self.B = nn.Parameter(torch.randn(opts['num_samples'] * opts['z_dim'], opts['z_dim']))

    def encode(self, x):
        mu, std = self.encoder.forward(x.view(-1, self.x_dim))

        #make B
        B = []
        for i in range(self.num_samples):
            u = self.B[i*self.z_dim : (i + 1)*self.z_dim, :]
            B.append(self.orthogonalize(u))
        B = torch.cat(B, 0)
        B = torch.cat([B, Variable(torch.zeros(self.num_samples * self.z_dim, (self.num_samples-1) * self.z_dim)).cuda()], 1)

        B = B.unsqueeze(0).repeat(mu.size(0), 1, 1)
        B = B * std.repeat(1, self.num_samples).unsqueeze(2)

        mu = mu.view(mu.size(0), -1, self.z_dim)
        std = std.view(mu.size(0), -1, self.z_dim)
        mu = mu.expand(-1, self.num_samples, self.z_dim)
        std = std.expand(-1, self.num_samples, self.z_dim)

        return mu, std, B

    def orthogonalize(self, A):
        s = torch.matmul(A, A.t()) + Variable(torch.eye(A.size(0))).cuda() * 1e-5
        l = torch.potrf(s, upper=False)
        u = torch.matmul(torch.inverse(l), A)
        u = torch.sqrt(1. / torch.sum(u * u, 1, keepdim=True)) * u
        return u

    def covariance_matrix(self):
        B = []
        for i in range(self.num_samples):
            u = self.B[i*self.z_dim : (i + 1)*self.z_dim, :]
            B.append(self.orthogonalize(u))
        B = torch.cat(B, 0)
        B = torch.cat([B, Variable(torch.zeros(self.num_samples * self.z_dim, (self.num_samples-1) * self.z_dim)).cuda()], 1)
        return torch.matmul(B, B.t())

class COVKA(COV):
    def __init__(self, opts):
        super(COVKA, self).__init__(opts)
        self.encoder = Encoder(opts['x_dim'], opts['h_dim'], opts['z_dim'])
        self.decoder = Decoder(opts['x_dim'], opts['h_dim'], opts['z_dim'])
        self.B = nn.Parameter(torch.randn(opts['num_samples'] , 1))

    def encode(self, x):
        mu, std = self.encoder.forward(x.view(-1, self.x_dim))

        # creating A = diag(std)
        A = torch.cat([std.unsqueeze(2),Variable(torch.zeros(std.size(0), std.size(1), std.size(1))).cuda()], 2)
        A = A.view(-1, std.size(1) + 1, std.size(1))
        A = A[:,:-1,:]

        # expanding B if necessary
        B = self.B
        B = torch.cat([B, Variable(torch.zeros(self.num_samples, self.num_samples-1)).cuda()], 1)
        B = B.unsqueeze(0)

        # normalizing rows of B
        B = torch.sqrt(1. / torch.sum(B*B, 2, keepdim=True)) * B

        L = kronecker_product(B, A)

        mu = mu.view(mu.size(0), -1, self.z_dim)
        std = std.view(mu.size(0), -1, self.z_dim)
        mu = mu.expand(-1, self.num_samples, self.z_dim)
        std = std.expand(-1, self.num_samples, self.z_dim)

        return mu, std, L

class COVKB(COV):
    def __init__(self, opts):
        super(COVKB, self).__init__(opts)
        self.encoder = Encoder(opts['x_dim'], opts['h_dim'], opts['z_dim'])
        self.decoder = Decoder(opts['x_dim'], opts['h_dim'], opts['z_dim'])
        self.B = nn.Parameter(torch.randn(opts['num_samples'] , opts['num_samples']))

    def encode(self, x):
        mu, std = self.encoder.forward(x.view(-1, self.x_dim))

        # creating A = diag(std)
        A = torch.cat([std.unsqueeze(2),Variable(torch.zeros(std.size(0), std.size(1), std.size(1))).cuda()], 2)
        A = A.view(-1, std.size(1) + 1, std.size(1))
        A = A[:,:-1,:]

        # expanding B if necessary
        B = self.B
        B = B.unsqueeze(0)

        # normalizing rows of B
        B = torch.sqrt(1. / torch.sum(B*B, 2, keepdim=True)) * B

        L = kronecker_product(B, A)

        mu = mu.view(mu.size(0), -1, self.z_dim)
        std = std.view(mu.size(0), -1, self.z_dim)
        mu = mu.expand(-1, self.num_samples, self.z_dim)
        std = std.expand(-1, self.num_samples, self.z_dim)

        return mu, std, L

class COVR4(COV):
    def __init__(self, opts):
        super(COVR4, self).__init__(opts)
        self.r = 4
        self.b = self.r * self.z_dim
        self.B = Variable(make_rotation_matrix(self.r, self.z_dim)).cuda()

class COVR6(COV):
    def __init__(self, opts):
        super(COVR6, self).__init__(opts)
        self.r = 6
        self.b = self.r * self.z_dim
        self.B = Variable(make_rotation_matrix(self.r, self.z_dim)).cuda()

class COVR(COV):
    def __init__(self, opts):
        super(COVR, self).__init__(opts)
        self.b = self.num_samples * self.z_dim
        self.B = Variable(make_rotation_matrix(self.num_samples, self.z_dim)).cuda()

def make_rotation_matrix(r, z_dim):
    a = 2 * np.pi / r
    R0 = torch.FloatTensor([[np.cos(a), -np.sin(a)],
                            [np.sin(a), np.cos(a)]])
    # R0 = torch.cat([R0, torch.zeros(2, z_dim-2)], 1)
    # T = torch.cat([torch.zeros(z_dim-2, 2), torch.eye(z_dim-2, z_dim-2)], 1)
    # R0 = torch.cat([R0, T], 0)
    n = z_dim/2
    R0 = kronecker_product(torch.eye(n), R0).squeeze()

    R = R0
    B = torch.eye(z_dim)
    for i in range(r-1):
        B = torch.cat([B, R], 0)
        R = torch.mm(R,R0)
    B = torch.cat([B, torch.zeros(z_dim*r, z_dim*r-z_dim)], 1)
    return B

def kronecker_product(B, A):
    # computes B \otimes A 
    a = A.size(1)
    b = B.size(1)
    A = A.repeat(1, b, b)

    B = B.contiguous()
    B = B.view(-1, b * b, 1)
    B = B.repeat(1, 1, a)
    B = B.view(-1, b, b * a)
    B = B.repeat(1, 1, a)
    B = B.view(-1, b * a, b * a)

    L = B * A
    return L


def fc_block(in_dim, out_dim, activation=True, dropout=False):
    modules = [
        nn.Linear(in_dim, out_dim),
    ]

    if dropout:
        modules.append(torch.nn.Dropout())

    if activation:
        modules += [
            nn.Tanh()
            # nn.ReLU()
        ]

    return nn.Sequential(*modules)

def make_net(x_dim, h_dim, z_dim):
    net = nn.Sequential(
            fc_block(x_dim, h_dim),
            fc_block(h_dim, h_dim),
            # fc_block(h_dim, z_dim),
        )
    return net

class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(Encoder, self).__init__()

        self.net = make_net(x_dim, h_dim, z_dim)
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        z = self.net.forward(x)
        return self.fc1(z), torch.exp(self.fc2(z))

class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(Decoder, self).__init__()

        self.net = make_net(z_dim, h_dim, x_dim)
        self.fc = nn.Linear(h_dim, x_dim)

    def forward(self, x):
        z = self.fc(self.net.forward(x))
        return torch.sigmoid(z)
