from __future__ import print_function
import argparse
import struct
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

class DictDataset(torch.utils.data.Dataset):
    def __init__(self, d, keys):
        n_examples = None
        for key in keys:
            if n_examples is None:
                n_examples = d[key].size(0)
            else:
                assert(d[key].size(0) == n_examples)

        self.data = [d[key] for key in keys]

    def __getitem__(self, index):
        return [v[index] for v in self.data]

    def __len__(self):
        return self.data[0].size(0)

class BinarizedDictDataset(torch.utils.data.Dataset):
    def __init__(self, d, keys):
        n_examples = None
        for key in keys:
            if n_examples is None:
                n_examples = d[key].size(0)
            else:
                assert(d[key].size(0) == n_examples)

        self.data = [d[key] for key in keys]

    def __getitem__(self, index):
        r = []
        for d in self.data:
            v = d[index]
            v = torch.le(torch.rand(v.size()), v).type(torch.FloatTensor)
            r.append(v)
        return r

    def __len__(self):
        return self.data[0].size(0)

def load_mnist(opts):
    kwargs = {'num_workers': 1, 'pin_memory': True} if opts['cuda'] else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,transform=transforms.ToTensor()),
        batch_size=opts['batch_size'], shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=opts['batch_size'], shuffle=True, **kwargs)

    return train_loader, test_loader

def load_binarized_mnist(opts):
    n_used_for_validation = 400
    data_dir = '../data/raw/'
    kwargs = {'num_workers': 1, 'pin_memory': True} if opts['cuda'] else {}

    def load_mnist_images_np(imgs_filename):
        with open(imgs_filename, 'rb') as f:
            f.seek(4)
            nimages, rows, cols = struct.unpack('>iii', f.read(12))
            dim = rows*cols

            images = np.fromfile(f, dtype=np.dtype(np.ubyte))
            images = (images/255.0).astype('float32').reshape((nimages, dim))

        return images

    train_data = load_mnist_images_np(data_dir + 'train-images-idx3-ubyte')
    test_data = load_mnist_images_np(data_dir + 't10k-images-idx3-ubyte')

    validation_data = train_data[-n_used_for_validation:]
    train_data = train_data[:-n_used_for_validation]

    train_loader = torch.utils.data.DataLoader(BinarizedDictDataset({ 'data': torch.Tensor(train_data) }, ['data']),
                                               batch_size=opts['batch_size'], shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(BinarizedDictDataset({ 'data': torch.Tensor(test_data) }, ['data']),
                                               batch_size=opts['batch_size'], shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(BinarizedDictDataset({ 'data': torch.Tensor(validation_data) }, ['data']),
                                               batch_size=opts['batch_size'], shuffle=True, **kwargs)
    return train_loader, test_loader, val_loader

def load_fixed_binarized_mnist(opts):
    data_dir = '../data/BinaryMNIST/'
    kwargs = {'num_workers': 1, 'pin_memory': True} if opts['cuda'] else {}

    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])

    with open(data_dir + 'binarized_mnist_train.amat') as f:
        lines = f.readlines()
    train_data = lines_to_np_array(lines).astype('float32')
    with open(data_dir + 'binarized_mnist_valid.amat') as f:
        lines = f.readlines()
    validation_data = lines_to_np_array(lines).astype('float32')
    with open(data_dir + 'binarized_mnist_test.amat') as f:
        lines = f.readlines()
    test_data = lines_to_np_array(lines).astype('float32')

    train_loader = torch.utils.data.DataLoader(DictDataset({ 'data': torch.Tensor(train_data) }, ['data']),
                                               batch_size=opts['batch_size'], shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(DictDataset({ 'data': torch.Tensor(test_data) }, ['data']),
                                               batch_size=opts['batch_size'], shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(DictDataset({ 'data': torch.Tensor(validation_data) }, ['data']),
                                               batch_size=opts['batch_size'], shuffle=True, **kwargs)
    return train_loader, test_loader, val_loader


# def load_skew_normal_1d(opts):
#     kwargs = {'num_workers': 1, 'pin_memory': True} if opts['cuda'] else {}
#     d = np.random.randn(10000, opts['x_dim'])
#     d = np.clip(d,-np.inf,0)
#     split = int(d.shape[0] * 0.8)
#     d_train = d[:split]
#     d_test = d[split:]

#     train_loader = torch.utils.data.DataLoader(DictDataset({ 'data': torch.Tensor(d_train) }, ['data']),
#                                                batch_size=opts['batch_size'], shuffle=True, **kwargs)
#     test_loader = torch.utils.data.DataLoader(DictDataset({ 'data': torch.Tensor(d_test) }, ['data']),
#                                                batch_size=opts['batch_size'], shuffle=True, **kwargs)
#     return train_loader, test_loader

