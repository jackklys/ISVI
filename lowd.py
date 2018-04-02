import argparse
import math
import sys
import os
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch.autograd import Variable
from torch import nn, optim
from scipy.stats import truncnorm
from experiments_1d import visualize_posterior as visualize_posterior_1d
from experiments_2d import visualize_posterior as visualize_posterior_2d

class GaussianMixture(nn.Module):
    def __init__(self, num_mixes, z_dim, mu, logvar):
        super(GaussianMixture, self).__init__()
        self.num_mixes = num_mixes
        self.mu = Variable(mu.expand(1, 1, z_dim, num_mixes))
        self.logvar = Variable(logvar.expand(1, 1, z_dim, num_mixes))
        self.pi = Variable(torch.ones(num_mixes))

        self.logsoftmax = nn.LogSoftmax(dim=0)
        self.softmax = nn.Softmax()

    def logprob(self, x):
        x = x.unsqueeze(3).expand(x.size(0), x.size(1), x.size(2), self.num_mixes)
        log_pi = self.logsoftmax(self.pi)
        log_pdf = - (0.5 * math.log(math.pi * 2) + 0.5 * self.logvar + ((x - self.mu) ** 2) / (2 * torch.exp(self.logvar))).sum(2)
        log_prob = log_sum_exp(log_pi + log_pdf, axis=2)
        return log_prob

    def sample(self, num_samples):
        pi_picked = torch.multinomial(self.pi, num_samples, replacement=True).view(-1)
        idx = torch.arange(0, num_samples).type(torch.LongTensor)

        sigma = torch.exp(0.5 * self.logvar).squeeze(1).repeat(num_samples, 1, 1)
        mu = self.mu.squeeze(1).repeat(num_samples, 1, 1)
        samples = torch.normal(mu, sigma)

        picked_samples = torch.transpose(samples, 1, 2)[idx, pi_picked, :]

        return picked_samples

class SliceGaussian(nn.Module):
    def __init__(self, z_dim, mu, logvar):
        super(SliceGaussian, self).__init__()
        self.mu = Variable(mu.expand(1, 1, z_dim))
        self.logvar = Variable(logvar.expand(1, 1, z_dim))     
        # self.mu = Variable(torch.zeros(1, 1, z_dim))
        # self.logvar = Variable(torch.zeros(1, 1, z_dim))
        self.epsilon = 0.01
        self.norm = 2 ** z_dim / (self.epsilon * (2 ** z_dim - 1) + 1)

    def heavyside(self, x):
        mask = torch.exp(torch.log(x.le(0).type(torch.FloatTensor)).sum(-1))
        mask = mask.type(torch.ByteTensor)
        h = Variable(torch.Tensor(x.size(0), x.size(1))).fill_(self.epsilon)
        h = h.masked_fill_(mask, 1)
        return h

    def log_normal(self, x):
        return -(((x - self.mu) ** 2) / (2 * torch.exp(self.logvar)) + 0.5 * self.logvar + 0.5 * np.log(2 * np.pi))

    def logprob(self, x):
        return (self.log_normal(x)).sum(2) + torch.log(self.heavyside(x)) + np.log(self.norm)

class HalfGaussian(nn.Module):
    def __init__(self, z_dim, mu, logvar):
        super(HalfGaussian, self).__init__()
        self.mu = Variable(mu.expand(1, 1, z_dim))
        self.logvar = Variable(logvar.expand(1, 1, z_dim))       
        self.epsilon = 0.01
        self.norm = 2 / (self.epsilon + 1)

    def heavyside(self, x):
        mask = x[:,:,0].le(0)
        h = Variable(torch.Tensor(x.size(0), x.size(1))).fill_(self.epsilon)
        h = h.masked_fill_(mask, 1)
        return h

    def log_normal(self, x):
        return -(((x - self.mu) ** 2) / (2 * torch.exp(self.logvar)) + 0.5 * self.logvar + 0.5 * np.log(2 * np.pi))

    def logprob(self, x):
        return (self.log_normal(x)).sum(2) + torch.log(self.heavyside(x)) + np.log(self.norm)

def log_sum_exp(v, axis=1):
    m = torch.max(v, axis)[0]
    r = torch.log(torch.sum(torch.exp(v - m.unsqueeze(axis)), axis)) + m
    return r

def log_mean_exp(v):
    m = torch.max(v, 1)[0]
    return torch.log(torch.mean(torch.exp(v - m.unsqueeze(1)), 1)) + m

class IWAE(nn.Module):
    def __init__(self, target, z_dim, num_samples, B=None, StdInit=False, FreezeMean=False):
        super(IWAE, self).__init__()
        self.p = target
        if StdInit:
            self.mu = nn.Parameter(torch.zeros(1, 1, z_dim))
            self.logstd = nn.Parameter(torch.zeros(1, 1, z_dim))        
        elif FreezeMean:
            self.mu = Variable(torch.zeros(1, 1, z_dim))
            self.logstd = nn.Parameter(torch.randn(1, 1, z_dim))
        else:
            self.mu = nn.Parameter(torch.randn(1, 1, z_dim))
            self.logstd = nn.Parameter(torch.randn(1, 1, z_dim))
        self.z_dim = z_dim
        self.num_samples = num_samples

    def encode(self, batch_size):
        mu = self.mu.expand(batch_size, self.num_samples, self.z_dim)
        logstd = self.logstd.expand(batch_size, self.num_samples, self.z_dim)
        return mu, logstd

    def sample(self, mu, logstd, ev=False):
        std = torch.exp(logstd)
        if ev:
            return mu
        else:
            eps = Variable(torch.normal(torch.zeros(std.size(0), self.num_samples, self.z_dim), std=1.0))
            return mu + std * eps

    def forward(self, batch_size, ev=False):
        mu, logstd = self.encode(batch_size)
        samples = self.sample(mu, logstd, ev)
        logvar =  2 * logstd
        return mu, logvar, samples

    def loss(self, batch_size):
        mu, logvar, samples = self.forward(batch_size)
        logq = - (((samples - mu) ** 2) / (2 * torch.exp(logvar)) + 0.5 * logvar + 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = self.p.logprob(samples)

        loss = log_mean_exp(logp - logq)
        loss = - torch.mean(loss)

        # w = logp - logq
        # log_w_tot = log_sum_exp(w, axis=1)
        # weights = torch.exp(w - log_w_tot.unsqueeze(1))
        # grad = torch.mean(torch.sum(weights * -samples.squeeze(2), 1))

        # print('correct_grad:' + str(grad))

        # mu_temp = self.mu.data
        # dloss = []
        # for h in [0.001, 0.00001, 0.0000001]:
        #     self.mu.data = mu_temp + h
        #     mu, logvar, samples = self.forward(batch_size)
        #     logq = - (((samples - mu) ** 2) / (2 * torch.exp(logvar)) + 0.5 * logvar + 0.5 * np.log(2 * np.pi)).sum(-1)
        #     logp = self.p.logprob(samples)

        #     loss1 = log_mean_exp(logp - logq)
        #     loss1 = - torch.mean(loss1)
        #     dloss.append((loss1 - loss) / h)

        # self.mu.data = mu_temp
        # self.derivative(batch_size)

        return loss

    def derivative(self, batch_size):
        mu, logvar, samples = self.forward(batch_size, ev=False)
        logq = - (((samples - mu) ** 2) / (2 * torch.exp(logvar)) + 0.5 * logvar + 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = self.p.logprob(samples)

        loss = log_mean_exp(logp - logq)
        loss = - torch.mean(loss)

        mu_temp = self.mu.data
        dloss = []
        for h in [0.1, 0.001, 0.00001]:
            self.mu.data = mu_temp + h
            mu, logvar, _ = self.forward(batch_size, ev=False)
            logq = - (((samples - mu) ** 2) / (2 * torch.exp(logvar)) + 0.5 * logvar + 0.5 * np.log(2 * np.pi)).sum(-1)
            logp = self.p.logprob(samples)

            loss1 = log_mean_exp(logp - logq)
            loss1 = - torch.mean(loss1)
            dloss.append((loss1 - loss) / h)

        self.mu.data = mu_temp

        loss.backward(retain_graph=True)
        print('mu:' + str(mu_temp))
        print('grad:' + str(self.mu.grad))
        print('derivative:' + str(dloss))
        return dloss

    def covariance_matrix(self):
        B = Variable(torch.eye(self.z_dim * self.num_samples))
        std = torch.exp(self.logstd).squeeze()
        std = std.repeat(self.num_samples)
        B = B * std.unsqueeze(1)
        cov = torch.matmul(B, B.t())
        return cov, B

class IWAE2(IWAE):
    def __init__(self, target, z_dim, num_samples, B=None, StdInit=False, FreezeMean=False):
        super(IWAE2, self).__init__(target, z_dim, num_samples, B, StdInit, FreezeMean)
        if StdInit:
            self.mu = nn.Parameter(torch.zeros(1, num_samples, z_dim))
            self.logstd = nn.Parameter(torch.zeros(1, num_samples, z_dim))
        elif FreezeMean:
            self.mu = Variable(torch.zeros(1, num_samples, z_dim))
            self.logstd = nn.Parameter(torch.randn(1, num_samples, z_dim))
        else:
            self.mu = nn.Parameter(torch.randn(1, num_samples, z_dim))
            self.logstd = nn.Parameter(torch.randn(1, num_samples, z_dim))

class IWAEN(IWAE):
    def __init__(self, target, z_dim, num_samples, B=None, StdInit=False, FreezeMean=False):
        super(IWAEN, self).__init__(target, z_dim, num_samples, B, StdInit, FreezeMean)

    def sample(self, mu, logstd, ev=False):
        std = torch.exp(logstd)
        if ev:
            return mu
        else:
            eps1 = Variable(torch.normal(torch.zeros(std.size(0), int(self.num_samples/2), self.z_dim), std=1.0))
            eps2 = -eps1
            eps = torch.cat([eps1, eps2], 1)
            return mu + std * eps

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

class COV(IWAE):
    def __init__(self, target, z_dim, num_samples, B=None, StdInit=False, FreezeMean=False):
        super(COV, self).__init__(target, z_dim, num_samples, B, StdInit, FreezeMean)
        if StdInit:
            self.mu = nn.Parameter(torch.zeros(1, z_dim))
            self.logstd = nn.Parameter(torch.zeros(z_dim))
        elif FreezeMean:
            self.mu = Variable(torch.zeros(1, z_dim))
            self.logstd = nn.Parameter(torch.randn(z_dim))
        else:
            self.mu = nn.Parameter(torch.randn(1, z_dim))
            self.logstd = nn.Parameter(torch.randn(z_dim))        

        # self.B = nn.Parameter(torch.eye(num_samples).unsqueeze(0))
        if B is not None: 
            self.B = Variable(B.unsqueeze(0))
        else:
            self.B = nn.Parameter(torch.randn(num_samples * z_dim, num_samples * z_dim).unsqueeze(0))
        # self.B = nn.Parameter(torch.FloatTensor([[1,0],[-1,0]]).unsqueeze(0))
        # self.B = nn.Parameter(torch.randn(2,2)).unsqueeze(0)
        print(self.B)

    def encode(self, batch_size):
        mu = self.mu.expand(batch_size, self.z_dim).repeat(1, self.num_samples)
        logstd = self.logstd.expand(batch_size, self.z_dim).repeat(1, self.num_samples)

        #make B
        b = self.B.size(1)
        B = self.B.expand(batch_size, b, b)
        B = torch.sqrt(1. / torch.sum(B*B, 2)).unsqueeze(2) * B
        n = self.num_samples * self.z_dim / b
        B = kronecker_product(Variable(torch.eye(n).expand(batch_size, n, n)), B)
        B = B * torch.exp(logstd).unsqueeze(2)

        return mu, logstd, B

    def sample(self, mu, A):
        eps = Variable(torch.normal(torch.zeros(A.size(0), self.num_samples * self.z_dim, 1), std=1.0))
        std = torch.matmul(A, eps)
        return mu.unsqueeze(2) + std

    def forward(self, batch_size):
        mu, logstd, L = self.encode(batch_size)
        samples = self.sample(mu, L).view(-1, self.num_samples, self.z_dim)

        mu = mu.view(-1, self.num_samples, self.z_dim)
        logstd = logstd.view(-1, self.num_samples, self.z_dim)
        logvar = 2 * logstd

        return mu, logvar, samples

    def covariance_matrix(self):
        B = self.B
        b = B.size(1)
        n = self.num_samples * self.z_dim / b
        B = kronecker_product(Variable(torch.eye(n).expand(1, n, n)), B)
        B = B.squeeze(0)
        B = torch.sqrt(1. / torch.sum(B*B, 1)).unsqueeze(1) * B
        std = torch.exp(self.logstd)
        std = std.repeat(self.num_samples)
        B = B * std.unsqueeze(1)
        cov = torch.matmul(B, B.t())
        return cov, B

def make_rotation_matrix(n, z_dim):
    r = n
    a = 2 * np.pi / r
    R0 = torch.FloatTensor([[np.cos(a), -np.sin(a)],
                            [np.sin(a), np.cos(a)]])

    R0 = kronecker_product(torch.eye(z_dim/2), R0).squeeze()

    R = R0
    B = torch.eye(z_dim)
    for i in range(r-1):
        B = torch.cat([B, R], 0)
        R = torch.mm(R,R0)
    B = torch.cat([B, torch.zeros(z_dim * r, z_dim * r - z_dim)], 1)
    return B

class COVR4(COV):
    def __init__(self, target, z_dim, num_samples, B=None, StdInit=False, FreezeMean=False):
        super(COVR4, self).__init__(target, z_dim, num_samples, B=make_rotation_matrix(4, z_dim), StdInit=StdInit, FreezeMean=FreezeMean)

class COVR6(COV):
    def __init__(self, target, z_dim, num_samples, B=None, StdInit=False, FreezeMean=False):
        super(COVR6, self).__init__(target, z_dim, num_samples, B=make_rotation_matrix(6, z_dim), StdInit=StdInit, FreezeMean=FreezeMean)

class COVD(COV):
    def __init__(self, target, z_dim, num_samples, B=None, StdInit=False, FreezeMean=False, FreezeAll=False):
        super(COVD, self).__init__(target, z_dim, num_samples, B, StdInit, FreezeMean)
        if StdInit:
            self.mu = nn.Parameter(torch.zeros(1, z_dim))
            self.logstd = nn.Parameter(torch.zeros(z_dim))
        elif FreezeMean:
            self.mu = Variable(torch.zeros(1, z_dim))
            self.logstd = nn.Parameter(torch.randn(z_dim))
        elif FreezeAll:
            self.mu = Variable(torch.zeros(1, z_dim))
            self.logstd = Variable(torch.zeros(z_dim))
        else:
            self.mu = nn.Parameter(torch.randn(1, z_dim))
            self.logstd = nn.Parameter(torch.randn(z_dim))        
        if B is not None:
            self.B = Variable(B.unsqueeze(0))
        else:
            self.B = nn.Parameter(torch.randn(num_samples * z_dim, z_dim))

    def encode(self, batch_size):
        mu = self.mu.expand(batch_size, self.z_dim).repeat(1, self.num_samples)
        logstd = self.logstd.expand(batch_size, self.z_dim).repeat(1, self.num_samples)

        #make B
        B = []
        for i in range(self.num_samples):
            u = self.B[i*self.z_dim : (i + 1)*self.z_dim, :]
            B.append(self.orthogonalize(u))
        B = torch.cat(B, 0)
        B = torch.cat([B, Variable(torch.zeros(self.num_samples * self.z_dim, (self.num_samples-1) * self.z_dim))], 1)

        b = B.size(1)
        B = B.expand(batch_size, b, b)
        B = B * torch.exp(logstd).unsqueeze(2)

        return mu, logstd, B

    def orthogonalize(self, A):
        s = torch.matmul(A, A.t()) + Variable(torch.eye(A.size(0))) * 1e-5
        l = torch.potrf(s, upper=False)
        u = torch.matmul(torch.inverse(l), A)
        u = torch.sqrt(1. / torch.sum(u*u, 1, keepdim=True)) * u
        return u

    def covariance_matrix(self):
        B = self.encode(1)[2].squeeze(0)
        return torch.matmul(B, B.t()), B

class COVF(COV):
    def __init__(self, target, z_dim, num_samples, B=None, StdInit=False, FreezeMean=False):
        super(COVF, self).__init__(target, z_dim, num_samples, B, StdInit, FreezeMean)
        if StdInit:
            self.mu = nn.Parameter(torch.zeros(1, z_dim * num_samples))
        elif FreezeMean:
            self.mu = Variable(torch.zeros(1, z_dim * num_samples))
        else:
            self.mu = nn.Parameter(torch.randn(1, z_dim * num_samples))
        self.B = nn.Parameter(torch.randn(z_dim * num_samples, z_dim * num_samples).unsqueeze(0))

    def encode(self, batch_size):
        mu = self.mu.repeat(batch_size, 1)
        B = self.B.repeat(batch_size, 1, 1)
        return mu, B

    def forward(self, batch_size):
        mu, L = self.encode(batch_size)
        samples = self.sample(mu, L).view(-1, self.num_samples, self.z_dim)
        mu = mu.view(-1, self.num_samples, self.z_dim)
        cov = torch.matmul(L[0,:,:], L[0,:,:].t())
        marginals = self.get_marginals(cov)
        return mu, marginals, samples

    def loss(self, batch_size):
        mu, marginals, samples = self.forward(batch_size)

        ico = []
        det = []
        for i in range(self.num_samples):
            m = marginals[i, :, :]
            ico.append(torch.inverse(m).unsqueeze(0))
            det.append(torch.potrf(m).diag().prod() ** 2)

        ico = torch.cat(ico, 0).repeat(batch_size, 1, 1)
        det = torch.cat(det).unsqueeze(0)
        y = (samples - mu).view(-1, self.z_dim, 1)

        a = torch.matmul(ico, y)
        z = torch.matmul(torch.transpose(y, 1, 2), a)
        z = z.view(-1, self.num_samples)

        logq = -0.5 * z - 0.5 * self.z_dim * np.log(2 * np.pi) - 0.5 * torch.log(det)
        logp = self.p.logprob(samples)

        loss = log_mean_exp(logp - logq)
        loss = - torch.mean(loss)
        return loss

    def get_marginals(self, cov):
        mask = kronecker_product(torch.eye(self.num_samples).unsqueeze(0), 
                    torch.FloatTensor(1, self.z_dim, self.z_dim).fill_(1)).type(torch.ByteTensor).squeeze(0)
        mask = Variable(mask)
        marginals = torch.masked_select(cov, mask).view(self.num_samples, self.z_dim, self.z_dim)
        return marginals

    def covariance_matrix(self):
        B = self.encode(1)[1].squeeze(0)
        return torch.matmul(B, B.t()), B

class COVF2(COVF):
    def __init__(self, target, z_dim, num_samples, B=None, StdInit=False, FreezeMean=False):
        super(COVF, self).__init__(target, z_dim, num_samples, B, StdInit, FreezeMean)
        if StdInit:
            self.mu = nn.Parameter(torch.zeros(1, z_dim))
            self.logstd = nn.Parameter(torch.zeros(z_dim))
        elif FreezeMean:
            self.mu = Variable(torch.zeros(1, z_dim))
            self.logstd = nn.Parameter(torch.randn(z_dim))
        else:
            self.mu = nn.Parameter(torch.randn(1, z_dim))
            self.logstd = nn.Parameter(torch.randn(z_dim))      
        self.B = nn.Parameter(torch.randn(z_dim * num_samples, z_dim).unsqueeze(0))

    def encode(self, batch_size):
        std = torch.exp(self.logstd)
        mu = self.mu.expand(batch_size, self.z_dim).repeat(1, self.num_samples)
        std = std.expand(batch_size, self.z_dim).repeat(1, self.num_samples)

        #make B
        B = torch.cat([self.B, Variable(torch.zeros(1, self.num_samples * self.z_dim, (self.num_samples - 1) * self.z_dim))], 2)
        B = B.repeat(batch_size, 1, 1)
        B = torch.sqrt(1. / torch.sum(B*B, 2)).unsqueeze(2) * B
        B = B * std.unsqueeze(2)
        return mu, B

    def covariance_matrix(self):
        B = torch.cat([self.B, Variable(torch.zeros(1, self.num_samples * self.z_dim, (self.num_samples - 1) * self.z_dim))], 2)
        B = B.squeeze(0)
        B = torch.sqrt(1. / torch.sum(B*B, 1)).unsqueeze(1) * B
        std = torch.exp(self.logstd)
        std = std.repeat(self.num_samples)
        B = B * std.unsqueeze(1)
        cov = torch.matmul(B, B.t())
        return cov, B

class COVK(COV):
    def __init__(self, target, z_dim, num_samples, B=None, StdInit=False, FreezeMean=False):
        super(COVK, self).__init__(target, z_dim, num_samples, B=B, StdInit=StdInit, FreezeMean=FreezeMean)
        self.b = num_samples
        self.B = nn.Parameter(torch.randn(self.b, self.b).unsqueeze(0))

    def encode(self, batch_size):
        B = self.B.expand(batch_size, self.b, self.b)
        A = torch.diag(torch.exp(self.logstd)).expand(batch_size, self.z_dim, self.z_dim)

        #make full B matrix
        n = self.num_samples /self.b
        B = kronecker_product(Variable(torch.eye(n).expand(batch_size, n, n)), B)
        B = torch.sqrt(1. / torch.sum(B*B, 2)).unsqueeze(2) * B

        L = kronecker_product(B, A)
        mu = self.mu.expand(batch_size, self.z_dim).repeat(1, self.num_samples)
        logstd = self.logstd.expand(batch_size, self.z_dim).repeat(1, self.num_samples)

        return mu, logstd, L

    def covariance_matrix(self):
        std = torch.exp(self.logstd)
        A = torch.diag(std).expand(1, self.z_dim, self.z_dim)
        B = self.B

        #make full B matrix
        n = self.num_samples /self.b
        B = kronecker_product(Variable(torch.eye(n).expand(1, n, n)), B)
        B = torch.sqrt(1. / torch.sum(B*B, 2)).unsqueeze(2) * B

        L = kronecker_product(B, A).squeeze()
        cov = torch.matmul(L, L.t())
        return cov, L

class COVK2(COVK):
    def __init__(self, target, z_dim, num_samples, B=None, StdInit=False, FreezeMean=False):
        super(COVK2, self).__init__(target, z_dim, num_samples, B=B, StdInit=StdInit, FreezeMean=FreezeMean)
        self.b = 2
        self.B = nn.Parameter(torch.randn(self.b, self.b).unsqueeze(0))

def run_train(model, epochs, optimizer, scheduler):
    loss_curve = []
    
    param_list = list(model.parameters())
    grads = []
    grad_vars = []
    for x in param_list:
        grads.append([[] for i in range(x.view(-1).size(0))])
        grad_vars = copy.deepcopy(grads)

    for i in range(epochs):
        optimizer.zero_grad()
        loss = model.loss(batch_size=100)
        loss.backward()

        # check grads
        # print('mu:' + str(model.mu))
        # print('grad:' + str(model.mu.grad))

        optimizer.step()
        # scheduler.step()
        if i%500==0:
            l = model.loss(batch_size=10000).data[0]
            print(l)
            print('mu: ' + str(model.mu.data))
            print('logstd: ' + str(model.logstd.data))
            # save_dir = 'results/{}d/{}/{}/{}/'.format(model.z_dim, model.num_samples, model.p.__class__.__name__, model.__class__.__name__)
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            # if model.z_dim==1 and model.__class__.__name__!='COVF':
            #     visualize_posterior_1d(model, save_dir  + str(i))
            # elif model.z_dim==2 and (model.__class__.__name__!='COVF' and model.__class__.__name__!='COVF2'):
            #     visualize_posterior_2d(model, save_dir  + str(i))

            # if hasattr(model, 'B'):
            #     print("cov:")
            #     print(model.covariance_matrix().data)
            loss_curve.append(l)

            for i, x in enumerate(param_list):
                j = 0
                for y in x.grad.data.view(-1):
                    grads[i][j].append(y)
                    grad_vars[i][j].append(np.var(np.array(grads[i][j])))
                    j += 1

    return loss_curve, grads, grad_vars