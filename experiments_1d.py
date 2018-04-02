import argparse
import math
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.utils.data
from torch.autograd import Variable
from torch import nn, optim
import lowd

def log_sum_exp(v, axis=1):
    m = torch.max(v, axis)[0]
    r = torch.log(torch.sum(torch.exp(v - m.unsqueeze(axis)), axis)) + m
    return r

def posterior_samples(model, batch_size=10000):
    mu, logvar, samples = model.forward(batch_size)
    # mu = mu.contiguous()
    # samples = samples.view(batch_size, model.num_samples, model.z_dim)
    # samples = samples.contiguous()
    # mu = mu.view(batch_size, model.num_samples, model.z_dim)
    # logvar = logvar.view(batch_size, model.num_samples, model.z_dim)

    logq = - (((samples - mu) ** 2) / (2 * torch.exp(logvar)) + 0.5 * logvar + 0.5 * np.log(2 * np.pi)).sum(-1)
    logp = model.p.logprob(samples)

    log_w_tot = log_sum_exp(logp - logq, axis=1)
    weights = torch.exp(logp - logq - log_w_tot.unsqueeze(1))

    picked_id = torch.multinomial(weights, 1, replacement=True).view(-1)

    idx = torch.arange(0, batch_size).type(torch.LongTensor)

    picked_samples = samples[idx, picked_id.data, :]
    picked_samples = picked_samples.data.numpy()
    return picked_samples

def visualize_posterior(model, save_dir):
    #plot target distribution
    x = np.linspace(-10, 10, 1000)
    z = Variable(torch.FloatTensor(x)).view(-1, 1, 1)
    y = model.p.logprob(z).view(-1)
    y = torch.exp(y)
    y = y.data.numpy()
    plt.plot(x, y, label='target', color='r')

    #plot posterior samples
    samples = posterior_samples(model)
    sns.distplot(samples, label='approximate')
    plt.legend()
    plt.savefig(save_dir + 'samples.png')
    plt.close()

def plot_loss(loss_curve, save_dir):
    plt.plot(loss_curve, color='red')
    plt.xlabel('epochs')
    plt.ylabel('negative log-likelihood')
    plt.savefig(save_dir + 'loss.png')
    plt.close()

def plot_grads(grads, save_dir):
    colors = ['red', 'blue', 'green']
    for i, x in enumerate(grads):
        for y in x:
            plt.plot(y, color=colors[i])
    plt.xlabel('epochs')
    plt.ylabel('gradient')
    plt.savefig(save_dir + 'grads.png')
    plt.close()

def plot_gradvars(gradvars, save_dir):
    colors = ['red', 'blue', 'green']
    for i, x in enumerate(gradvars):
        for y in x:
            plt.plot(y, color=colors[i])
    plt.xlabel('epochs')
    plt.ylabel('variance of gradient')
    plt.savefig(save_dir + 'gradvars.png')
    plt.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='IWAE')
    parser.add_argument('-k', type=int, default=12)
    parser.add_argument('-target', type=str, default='SliceGaussian')
    args = parser.parse_args()
    model_name = args.model
    target = args.target
    k = args.k

    #load model
    print(model_name)
    load_dir = 'results/1d/' + str(k) +'/'+target+'/' +model_name + '/'
    model = torch.load(load_dir + 'model')

    visualize_posterior(model, load_dir)
