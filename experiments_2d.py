import argparse
import math
import sys
import os
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
import cPickle as pkl
from model_list import all_models


def log_sum_exp(v, axis=1):
    m = torch.max(v, axis)[0]
    r = torch.log(torch.sum(torch.exp(v - m.unsqueeze(axis)), axis)) + m
    return r

def posterior_samples(model, batch_size=10000):
    mu, logvar, samples = model.forward(batch_size)
    mu = mu.contiguous()
    samples = samples.view(batch_size, model.num_samples, model.z_dim)
    samples = samples.contiguous()
    mu = mu.view(batch_size, model.num_samples, model.z_dim)
    logvar = logvar.view(batch_size, model.num_samples, model.z_dim)

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
    sns.set_style('darkgrid')
    sns.despine(left=True, bottom=True)
    cmap = sns.dark_palette("red", as_cmap=True)

    # plot target distribution
    n = 100
    x = np.linspace(-5, 5, n)
    y = np.linspace(-5, 5, n)
    xv, yv = np.meshgrid(x, y)
    zv = np.concatenate((xv.reshape(-1,1), yv.reshape(-1,1)), axis=1)
    zv = Variable(torch.FloatTensor(zv).view(-1, 1, 2))
    z = model.p.logprob(zv).view(n, n)
    z = torch.exp(z)
    z = z.data.numpy()
    plt.contour(x, y, z, colors='r', linewidths='2')
    sns.set_context(rc={"lines.linewidth": 2})

    #plot target distribution
    # samples = model.p.sample(n).data.numpy()   
    # sns.kdeplot(samples[:,0], samples[:,1], colors='red', cmap=None)
    # sns.kdeplot(samples[:,0], samples[:,1], cmap='Reds', shade=True, shade_lowest=False)
    # sns.kdeplot(samples[:,0], samples[:,1], n_levels=15, colors='red', cmap=None)

    #plot posterior samples
    n = 1000
    samples = posterior_samples(model, batch_size=n)
    cmap = sns.dark_palette("blue", as_cmap=True)
    # sns.kdeplot(samples[:,0], samples[:,1], cmap='Blues', shade=True, shade_lowest=False)

    sns.kdeplot(samples[:,0], samples[:,1], cmap=cmap)
    # plt.scatter(samples[:,0], samples[:,1], c="blue", s=30, linewidth=1, marker="+")
    # plt.legend()

    plt.savefig(save_dir + 'samples.png')
    plt.close()

def visualize_samples(model):
    sns.set_style('darkgrid')
    sns.despine(left=True, bottom=True)
    cmap = sns.dark_palette("red", as_cmap=True)
    
    n = 1000
    batch_size = 1

    # #plot target distribution
    # samples = model.p.sample(np).data.numpy()   
    # sns.kdeplot(samples[:,0], samples[:,1], cmap='Reds', shade=False, shade_lowest=False)

    # plot target distribution
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    xv, yv = np.meshgrid(x, y)
    zv = np.concatenate((xv.reshape(-1,1), yv.reshape(-1,1)), axis=1)
    zv = Variable(torch.FloatTensor(zv).view(-1, 1, 2))
    z = model.p.logprob(zv).view(n, n)
    z = torch.exp(z)
    z = z.data.numpy()
    plt.contour(x, y, z, colors='r')

    #plot posterior samples
    mu, logvar, samples = model.forward(batch_size)
    samples = samples.view(-1, model.z_dim)
    cmap = sns.dark_palette("blue", as_cmap=True)
    plt.scatter(samples[:,0], samples[:,1], c="blue", s=30, linewidth=1, marker="+")

def make_figure(save_dir):
    plt.figure(1)
    sns.set_style('darkgrid')

    model = torch.load(save_dir + 'model')

    plt.title(model.__class__.__name__)

    ax = plt.subplot(3,1,1, adjustable='box')
    visualize_samples(model)
    ax.set_aspect('equal')

    ax = plt.subplot(3,1,2)
    # plt.title(model.__class__.__name__)
    ax.grid(False)
    plt.imshow(model.covariance_matrix()[0].data, interpolation='none', vmin=-5, vmax=5, cmap='hot')

    ax = plt.subplot(3,1,3)
    ax.grid(False)
    plt.imshow(model.covariance_matrix()[1].data, interpolation='none', vmin=-5, vmax=5, cmap='hot')

    plt.savefig(save_dir + 'covmatrix.png')
    plt.close()

def make_figure_all(save_dir, models=all_models): 
    temp = []
    for m in models:
        if not os.path.exists(save_dir + m): 
            temp.append(m)
    for m in temp:
        models.remove(m)

    fig = plt.figure(figsize=(6,3))
    fig = plt.figure(1)

    sns.set_style('darkgrid')
    n = len(models)
    for i, m in enumerate(models):
        model = torch.load(save_dir + m + '/model')

        ax = plt.subplot(2,n,i+1)
        # ax.tick_params(labelbottom='off')   
        # ax.tick_params(labelleft='off')  
        # plt.title(model.__class__.__name__)
        visualize_samples(model)
        ax.set_aspect('equal')
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        ax.xaxis.set_major_formatter(plt.NullFormatter())


        ax = plt.subplot(2,n,n+i+1)
        ax.set_aspect('equal')
        ax.grid(False)
        ax.axis('off')
        plt.imshow(model.covariance_matrix()[0].data, interpolation='nearest', vmin=-2, vmax=2)

        # ax = plt.subplot(3,n,2*n+(i+1))
        # ax.grid(False)
        # plt.imshow(model.covariance_matrix()[1].data, interpolation='none', vmin=-5, vmax=5, cmap='hot')
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(save_dir + 'covmatrix.pdf', bbox_inches='tight')
    plt.savefig(save_dir + 'covmatrix.png', bbox_inches='tight')
    plt.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='IWAE')
    parser.add_argument('-k', type=int, default=12)
    parser.add_argument('-target', type=str, default='SliceGaussian')
    args = parser.parse_args()
    k = args.k
    target = args.target
    model_name = args.model

    make_figure_all('results/2d/' + str(k) + '/' + target + '/')