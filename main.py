import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch import nn, optim
from data import load_binarized_mnist
from engine import run_train
import cPickle as pkl
import math
import os
import sys
from models1 import IWAE, IWAEN, COV, COVR, COVR4, COVR6, COVD, COVKA, COVKB
from models2 import IWAE2
from models3 import COVK, COVK2, COVK3


parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='IWAE')
parser.add_argument('-data', type=str, default='mnist')
parser.add_argument('-k', type=int, default=12)
parser.add_argument('-e', type=int, default=10)
#3280

args = parser.parse_args()
directory = os.getcwd()

opts = {'batch_size': 20,
        'epochs': args.e,
        'log_interval': 500,
        'cuda': True,
        'x_dim': 784,
        'h_dim': 200,
        'z_dim': 10,
        'num_samples': args.k,
        }

# k = int(math.log(opts['epochs'])/math.log(3))
k = 7
milestones = [3**(i) for i in range(k)]
gamma = 0.1 ** (1./k)

if args.data == 'mnist':
        train_loader, test_loader, val_loader = load_binarized_mnist(opts)

opts['train_loader'] = train_loader
opts['test_loader'] = test_loader

torch.manual_seed(1)
if opts['cuda']: torch.cuda.manual_seed(1)

opts['model'] = getattr(sys.modules[__name__], args.model)(opts)
if opts['cuda']: opts['model'].cuda()

optimizer = optim.Adam(opts['model'].parameters(), eps=1e-4, lr=1e-3)
opts['optimizer'] = optimizer

scheduler = optim.lr_scheduler.MultiStepLR(opts['optimizer'], milestones=milestones, gamma=gamma)
opts['scheduler'] = scheduler

save_dir = 'results/mnist/' + str(opts['num_samples']) +'/'+ args.model + '/'
opts['save_dir'] = save_dir
if not os.path.exists(save_dir):
        os.makedirs(save_dir)

loss_curve = run_train(opts)
with open(save_dir + 'loss.pkl', 'wb') as f:
        pkl.dump(loss_curve, f)

plt.plot(loss_curve)
plt.savefig(save_dir + 'loss.png')
plt.close()

torch.save(opts['model'], save_dir + 'model')

# opts['train_loader'] = train_loader
# opts['test_loader'] = test_loader

# torch.manual_seed(1)
# if opts['cuda']: torch.cuda.manual_seed(1)

# if args.model == 'VAECOV':
# 	opts['model'] = getattr(models2, args.model)(opts)
# elif args.model == 'IWAE':
# 	opts['model'] = getattr(models, args.model)(opts)
	

# if opts['cuda']: opts['model'].cuda()

# optimizer = optim.Adam(opts['model'].parameters(), lr=1e-3)
# opts['optimizer'] = optimizer

# scheduler = optim.lr_scheduler.MultiStepLR(opts['optimizer'], milestones=milestones, gamma=gamma)
# opts['scheduler'] = scheduler

# plt.plot(run_train(opts), color='red') 

# plt.savefig(args.model + '_loss.png')
# plt.close()


# args.model = 'VAECOV'
# opts['train_loader'] = train_loader
# opts['test_loader'] = test_loader

# torch.manual_seed(1)
# if opts['cuda']: torch.cuda.manual_seed(1)

# opts['model'] = getattr(models2, args.model)(opts)
# if opts['cuda']: opts['model'].cuda()

# optimizer = optim.Adam(opts['model'].parameters(), lr=1e-3)
# opts['optimizer'] = optimizer

# plt.plot(run_train(opts), color='red') 


# args.model = 'IWAE'
# opts['train_loader'] = train_loader
# opts['test_loader'] = test_loader

# torch.manual_seed(1)
# if opts['cuda']: torch.cuda.manual_seed(1)

# opts['model'] = getattr(models, args.model)(opts)
# if opts['cuda']: opts['model'].cuda()

# optimizer = optim.Adam(opts['model'].parameters(), lr=1e-3)
# opts['optimizer'] = optimizer

# scheduler = optim.lr_scheduler.MultiStepLR(opts['optimizer'], milestones=milestones, gamma=gamma)
# opts['scheduler'] = scheduler

# plt.plot(run_train(opts), color='blue')

