from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# def adjust_learning_rate(optimizer, epoch):
#      lr = args.lr * (0.1 ** (epoch // 30))
#      for param_group in optimizer.param_groups:
#          param_group['lr'] = lr

def train(opts, epoch):
    opts['model'].train()
    train_loss = 0
    for batch_idx, data in enumerate(opts['train_loader']):
        data = data[0]
        data = Variable(data)
        if opts['cuda']:
            data = data.cuda()
        opts['optimizer'].zero_grad()
        loss, recon_x = opts['model'].loss(data)
        loss.backward()
        train_loss += loss.data[0] * len(data)
        opts['optimizer'].step()
        if batch_idx % opts['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(opts['train_loader'].dataset),
                100. * batch_idx / len(opts['train_loader']),
                loss.data[0] ))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(opts['train_loader'].dataset)))

def test(opts, epoch, loss_curve):
    opts['model'].eval()
    test_loss = 0
    for i, data in enumerate(opts['test_loader']):
        data = data[0]
        if opts['cuda']:
            data = data.cuda()
        data = Variable(data, volatile=True)
        loss, recon_x = opts['model'].loss(data)
        test_loss += loss.data[0] * len(data)
        # if opts['x_dim']==784:
        #     if i == 0:
        #       n = min(data.size(0), 8)
        #       comparison = torch.cat([data.view(data.size(0), 1, 28, 28)[:n],
        #                               recon_x.view(opts['batch_size'], 1, 28, 28)[:n]])
        #       save_image(comparison.data.cpu(),
        #                  'results/'+ opts['model'].__class__.__name__ +'/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss = test_loss / len(opts['test_loader'].dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    loss_curve.append(test_loss)
    if opts['model'].__class__.__name__=='COVD':
        print(opts['model'].covariance_matrix())

def run_train(opts):
    loss_curve = []
    for epoch in range(1, opts['epochs'] + 1):
        opts['scheduler'].step()
        # adjust_learning_rate(opts['optimizer'], epoch)
        train(opts, epoch)
        test(opts, epoch, loss_curve)

        if epoch % 10 == 0:
            # if opts['x_dim']==784:
            #     sample = Variable(torch.randn(64, opts['z_dim']))
            #     if opts['cuda']:
            #        sample = sample.cuda()
            #     sample = opts['model'].decode(sample).cpu()
            #     save_image(sample.data.view(64, 1, 28, 28),
            #                opts['save_dir'] + 'sample_' + str(epoch) + '.png')
            plt.plot(loss_curve)
            plt.savefig(opts['save_dir'] + 'loss.png')
            plt.close()

            if opts['model'].__class__.__name__=='COVD':
                plt.figure(1)
                ax = plt.subplot(111)
                plt.title(opts['model'].__class__.__name__)
                ax.grid(False)
                plt.imshow(opts['model'].covariance_matrix().data, interpolation='none')
                plt.savefig(opts['save_dir'] + 'covmatrix.png')
                plt.close()

    print(opts['model'].__class__.__name__)
    print(opts['num_samples'])
    return loss_curve

    