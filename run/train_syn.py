import os
import sys
sys.path.insert(0, './')
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn

from util.models import MLP
from util.dataset import load_pkl
from util.evaluation import accuracy
from util.device_parser import config_visible_gpu
from util.param_parser import IntListParser, DictParser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type = str, default = None,
        help = 'the file containing the training data')
    parser.add_argument('--batch_size', type = int, default = 100,
        help = 'the batch size, default = 100')
    parser.add_argument('--in_dim', type = int, default = 2,
        help = 'the number of input dimensions, default = 2')
    parser.add_argument('--hidden_dims', action = IntListParser, default = [],
        help = 'the number of neurons in hidden layers, default = []')
    parser.add_argument('--out_dim', type = int, default = 10,
        help = 'the number of output dimension, default = 10')
    parser.add_argument('--nonlinearity', type = str, default = 'relu',
        help = 'the activation function, default = "relu"')

    parser.add_argument('--total_iters', type = int, default = 100000,
        help = 'the number of total iterations, default = 100000')

    parser.add_argument('--optim', type = str, default = 'adam',
        help = 'the type of the optimizer, default = "adam"')
    parser.add_argument('--lr', type = float, default = 1e-4,
        help = 'the learning rate, default = 1e-4')

    parser.add_argument('--gpu', type = str, default = '0',
        help = 'choose which gpu to use, default = "0"')
    parser.add_argument('--out_folder', type = str, default = None,
        help = 'the output folder, default = None')
    parser.add_argument('--model_name', type = str, default = None,
        help = 'the name of the model')

    args = parser.parse_args()
    config_visible_gpu(args.gpu)
    if not args.gpu in ['cpu',] and torch.cuda.is_available():
        device = torch.device('cuda:0')
        device_ids = 'cuda'
        use_gpu = True
    else:
        device = torch.device('cpu')
        device_ids = 'cpu'
        use_gpu = False

    if args.data is None:
        raise ValueError('you should specify the input data')
    if args.out_folder is None:
        raise ValueError('you should specify the output folder')
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    model = MLP(in_dim = args.in_dim, hidden_dims = args.hidden_dims, out_dim = args.out_dim, nonlinearity = args.nonlinearity)
    criterion = nn.CrossEntropyLoss()
    model = model.cuda() if use_gpu else model
    criterion = criterion.cuda() if use_gpu else criterion

    if args.optim.lower() in ['sgd',]:
        optim = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 1e-4)
    elif args.optim.lower() in ['adam',]:
        optim = torch.optim.Adam(model.parameters(), lr = args.lr, betas = (0.9, 0.99), weight_decay = 1e-4)
    else:
        raise ValueError('Unrecognized Optimizer: %s' % args.optim.lower())

    data_loader = load_pkl(args.data, batch_size = args.batch_size)
    setup_config = {kwarg: value for kwarg, value in args._get_kwargs()}

    for idx in range(args.total_iters):

        data_batch, label_batch = next(data_loader)

        data_batch = data_batch.cuda(device) if use_gpu else data_batch
        label_batch = label_batch.cuda(device) if use_gpu else label_batch

        logits = model(data_batch)
        loss = criterion(logits, label_batch)
        acc = accuracy(logits.data, label_batch)

        optim.zero_grad()
        loss.backward()
        optim.step()

        sys.stdout.write('iter %d: accuracy = %.2f%%\r' % (idx, acc * 100.))

    pickle.dump(setup_config, open(os.path.join(args.out_folder, '%s.pkl' % args.model_name), 'wb'))
    torch.save(model.state_dict(), os.path.join(args.out_folder, '%s.ckpt' % args.model_name))

