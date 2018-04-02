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

def make_figure(save_dir):
    plt.figure(1)
    sns.set_style('darkgrid')

    model = torch.load(save_dir + 'model')

    ax = plt.subplot(2,1,1)
    plt.title(model.__class__.__name__)
    ax.grid(False)
    plt.imshow(model.covariance_matrix()[0].data, interpolation='none', vmin=-5, vmax=5, cmap='hot')

    ax = plt.subplot(2,1,2)
    ax.grid(False)
    plt.imshow(model.covariance_matrix()[1].data, interpolation='none', vmin=-5, vmax=5, cmap='hot')

    plt.savefig(save_dir + 'covmatrix.png')
    plt.close()

def make_figure_all(save_dir, models=None): 
    if not models:
        models = all_models

    temp = []
    for m in models:
        if not os.path.exists(save_dir + m): 
            temp.append(m)
    for m in temp:
        models.remove(m)

    plt.figure(1)
    sns.set_style('darkgrid')
    n = len(models)
    for i, m in enumerate(models):
        model = torch.load(save_dir + m + '/model')

        ax = plt.subplot(2,n,i+1)
        plt.title(model.__class__.__name__)
        ax.grid(False)
        ax.axis('off')
        ax.matshow(model.covariance_matrix()[0].data, interpolation='none', vmin=-5, vmax=5, cmap='hot')
        ax = plt.subplot(2,n,n+(i+1))
        ax.grid(False)
        ax.axis('off')

        ax.matshow(model.covariance_matrix()[1].data, interpolation='none', vmin=-5, vmax=5, cmap='hot')

    plt.savefig(save_dir + 'covmatrix.png')
    plt.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='IWAE')
    parser.add_argument('-k', type=int, default=12)
    parser.add_argument('-d', type=int, default=10)
    parser.add_argument('-target', type=str, default='HalfGaussian')
    args = parser.parse_args()
    k = args.k
    z_dim = args.d
    target = args.target
    model_name = args.model

    make_figure_all(save_dir = 'results/{}d/{}/{}/'.format(z_dim, k, target))
    # save_dir = 'results/{}d/{}/{}/{}/'.format(z_dim, k, target, model_name)
    # with open(save_dir + 'loss.pkl', 'rb') as f:
    #     p = pkl.load(f)
    # loss_curve = p[0]
    # grad_curves = p[1]

    # plt.plot(loss_curve, color='red')
    # plt.xlabel('epochs')
    # plt.ylabel('negative log-likelihood')
    # plt.savefig(save_dir + 'loss.png')
    # plt.close()

    # for x in grad_curves:
    #         plt.plot(x)
    # plt.savefig(save_dir + 'grads.png')
    # plt.close()


