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
from lowd import make_rotation_matrix
from experiments_1d import plot_loss, plot_grads, plot_gradvars
from experiments_2d import visualize_posterior, visualize_samples
from experiments_10d import make_figure
import cPickle as pkl


def train_gaussianmixture():
    # mu_target = torch.FloatTensor([[-1,3],
    #                                 [0,3]])
    # logvar_target = torch.FloatTensor([[-2,1],
    #                                     [-1,-2]])    
    # B = torch.eye(num_samples)

    torch.manual_seed(123)
    mu_target = torch.cat([5 * torch.rand(z_dim, 1) - 1, 5 * torch.rand(z_dim, 1) + 2], 1)
    logvar_target = torch.cat([-2 * torch.rand(z_dim, 1), torch.rand(z_dim, 1)], 1)
    print(mu_target)
    print(logvar_target)
    # mu_target = torch.FloatTensor([0])
    # logvar_target = torch.FloatTensor([0])
    target = lowd.GaussianMixture(num_mixes, z_dim, mu_target, logvar_target)
    model = getattr(lowd, model_name)(target, z_dim, num_samples)
    train(model)

def train_standardgaussian():
    # mu_target = torch.FloatTensor([[-1,3],
    #                                 [0,3]])
    # logvar_target = torch.FloatTensor([[-2,1],
    #                                     [-1,-2]])    
    # B = torch.eye(num_samples)

    torch.manual_seed(123)
    # mu_target = torch.cat([5 * torch.rand(10, 1) - 1, 5 * torch.rand(10, 1) + 5], 1)
    # logvar_target = torch.cat([-2 * torch.rand(10, 1), torch.rand(10, 1)], 1)
    # print(mu_target)
    # print(logvar_target)
    mu_target = torch.FloatTensor([0])
    logvar_target = torch.FloatTensor([0])
    target = lowd.GaussianMixture(num_mixes, z_dim, mu_target, logvar_target)
    model = getattr(lowd, model_name)(target, z_dim, num_samples)
    train(model)

def train_slicegaussian():
    mu_target = torch.zeros(z_dim)
    logvar_target = torch.zeros(z_dim)
    target = lowd.SliceGaussian(z_dim, mu_target, logvar_target)
    model = getattr(lowd, model_name)(target, z_dim, num_samples, FreezeMean=True)
    train(model)

def train_halfgaussian():
    mu_target = torch.zeros(z_dim)
    logvar_target = torch.zeros(z_dim)
    target = lowd.HalfGaussian(z_dim, mu_target, logvar_target)
    model = getattr(lowd, model_name)(target, z_dim, num_samples, FreezeMean=True)
    train(model)

def train(model):
    k = int(math.log(epochs)/math.log(3))
    milestones = [3**(i+1) for i in range(k)]
    gamma = 0.1 ** (1./k)
    optimizer = optim.Adam(model.parameters(), lr=(1e-4)/3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    print(model.__class__.__name__)
    print(model.p.__class__.__name__)
    print('num_samples: ' + str(num_samples))

    loss_curve, grad_curves, gradvar_curves = lowd.run_train(model, epochs, optimizer, scheduler)

    print(model.__class__.__name__)
    print(model.p.__class__.__name__)   
    print('num_samples: ' + str(num_samples))

    save_dir = 'results/{}d/{}/{}/{}/'.format(z_dim, num_samples, model.p.__class__.__name__, model.__class__.__name__)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model, save_dir + 'model')

    with open(save_dir +'loss.pkl', 'wb') as f:
        pkl.dump([loss_curve, grad_curves, gradvar_curves], f)
    with open(save_dir + 'loss.txt', 'w') as f:
        f.write(str(loss_curve))
        f.write(str([np.var(np.array(y)) for y in grad_curves]))

    plot_loss(loss_curve, save_dir)
    plot_grads(grad_curves, save_dir)
    plot_gradvars(gradvar_curves, save_dir)
    make_figure(save_dir)

parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='IWAE')
parser.add_argument('-k', type=int, default=12)
parser.add_argument('-e', type=int, default=120000)
parser.add_argument('-d', type=int, default=10)
args = parser.parse_args()
model_name = args.model
num_samples = args.k
z_dim = args.d
epochs = args.e 

num_mixes = 2

# train_gaussianmixture()
train_halfgaussian()
# train_slicegaussian()
# make_figure(12)





