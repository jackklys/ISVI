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
from lowd import log_sum_exp
from experiments_1d import visualize_posterior, plot_loss
import cPickle as pkl

def train_gaussianmixture():
    mu_target = torch.FloatTensor([-1, 0.3])
    logvar_target = torch.FloatTensor([-1, 2])
    # mu_target = torch.FloatTensor([0])
    # logvar_target = torch.FloatTensor([0])
    target = lowd.GaussianMixture(num_mixes, z_dim, mu_target, logvar_target)
    model = getattr(lowd, model_name)(target, z_dim, num_samples)
    train(model)

def train_halfgaussian():
    mu_target = torch.FloatTensor([0])
    logvar_target = torch.FloatTensor([0])
    target = lowd.HalfGaussian(z_dim, mu_target, logvar_target)
    model = getattr(lowd, model_name)(target, z_dim, num_samples, FreezeMean=True)
    train(model)

def train(model):    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    k = int(math.log(epochs)/math.log(3))
    milestones = [3**(i+1) for i in range(k)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    print(model.__class__.__name__)
    print(model.p.__class__.__name__)
    print('num_samples: ' + str(num_samples))

    loss_curve, grad_curves, gradvar_curves = lowd.run_train(model, epochs, optimizer, scheduler)
    print(model.__class__.__name__)

    save_dir = 'results/1d/' + str(num_samples) +'/'+ model.p.__class__.__name__ +'/'+ model.__class__.__name__ + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model, save_dir + 'model')

    with open(save_dir +'loss.pkl', 'wb') as f:
        pkl.dump(loss_curve, f)

    plot_loss(loss_curve, save_dir)
    visualize_posterior(model, save_dir)


parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='IWAE')
parser.add_argument('-k', type=int, default=12)
args = parser.parse_args()
model_name = args.model
num_samples = args.k

num_mixes = 2
z_dim = 1

# train_gaussianmixture()
train_halfgaussian()


