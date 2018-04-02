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
from experiments_2d import visualize_posterior, visualize_samples, make_figure
import cPickle as pkl


def train_gaussianmixture():
    mu_target = torch.FloatTensor([[-1,3],
                                    [0,3]])
    logvar_target = torch.FloatTensor([[-2,1],
                                        [-1,-2]])    
    # B = torch.eye(num_samples)
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
    mu_target = torch.FloatTensor([0])
    logvar_target = torch.FloatTensor([0])
    target = lowd.GaussianMixture(num_mixes, z_dim, mu_target, logvar_target)
    model = getattr(lowd, model_name)(target, z_dim, num_samples, FreezeMean=False)
    train(model)

def train_slicegaussian():
    mu_target = torch.FloatTensor([0, 0])
    logvar_target = torch.FloatTensor([0, 0])
    target = lowd.SliceGaussian(z_dim, mu_target, logvar_target)
    model = getattr(lowd, model_name)(target, z_dim, num_samples, FreezeMean=True)
    train(model)

def train_halfgaussian():
    mu_target = torch.FloatTensor([0, 0])
    logvar_target = torch.FloatTensor([0, 0])
    target = lowd.HalfGaussian(z_dim, mu_target, logvar_target)
    model = getattr(lowd, model_name)(target, z_dim, num_samples, FreezeMean=True)
    # B = np.linalg.cholesky(np.array([[1,-0.9999],[-0.9999,1]]))
    # B = torch.FloatTensor(B)
    # model = getattr(lowd, model_name)(target, z_dim, num_samples, B=B, FreezeAll=True)
    train(model)

def train(model):
    k = int(math.log(epochs)/math.log(3))
    milestones = [3**(i+1) for i in range(k)]
    gamma = 0.1 ** (1./k)
    optimizer = optim.Adam(model.parameters(), lr=((1e-4)/3))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


    print(model.__class__.__name__)
    print(model.p.__class__.__name__)
    print('num_samples: ' + str(num_samples))

    loss_curve, grad_curves, gradvar_curves  = lowd.run_train(model, epochs, optimizer, scheduler)
    print(model.__class__.__name__)
    print(model.p.__class__.__name__)
    print('num_samples: ' + str(num_samples))

    save_dir = 'results/2d/' + str(num_samples) +'/'+ model.p.__class__.__name__ +'/'+ model.__class__.__name__ + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model, save_dir + 'model')

    with open(save_dir +'loss.pkl', 'wb') as f:
        pkl.dump([loss_curve, grad_curves, gradvar_curves], f)
    with open(save_dir + 'loss.txt', 'w') as f:
        f.write(str(loss_curve)+'\n')
        f.write(str([np.var(np.array(y)) for y in grad_curves]))

    plot_loss(loss_curve, save_dir)
    plot_grads(grad_curves, save_dir)
    plot_gradvars(gradvar_curves, save_dir)
    plt.close()

    if model.__class__.__name__ not in ['COVF', 'COVF2']:
        visualize_posterior(model, save_dir)
    make_figure(save_dir)

parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='IWAE')
parser.add_argument('-k', type=int, default=2)
parser.add_argument('-e', type=int, default=120000)
args = parser.parse_args()
model_name = args.model
num_samples = args.k
epochs = args.e 

# num_samples = 10
num_mixes = 2
z_dim = 2

# train_gaussianmixture()
# train_standardgaussian()
train_halfgaussian()
# train_slicegaussian()
# make_figure(12)





