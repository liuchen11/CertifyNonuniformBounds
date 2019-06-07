import os
import sys
sys.path.insert(0, './')
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn

from util.models import MLP
from util.attack import PGM
from util.dataset import load_mnist, load_fmnist, load_svhn
from util.evaluation import accuracy, AverageCalculator
from util.device_parser import config_visible_gpu
from util.param_parser import IntListParser, DictParser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type = int, default = 100,
        help = 'the batch size, default = 100')
    parser.add_argument('--in_dim', type = int, default = 784,
        help = 'the number of input dimensions, default = 784')
    parser.add_argument('--hidden_dims', action = IntListParser, default = [],
        help = 'the number of neurons in hidden layers, default = []')
    parser.add_argument('--out_dim', type = int, default = 10,
        help = 'the number of classes, default = 10')
    parser.add_argument('--nonlinearity', type = str, default = 'relu',
        help = 'the activation function, default = "relu"')

    parser.add_argument('--epochs', type = int, default = 100,
        help = 'the number of epochs, default = 100')
    parser.add_argument('--dataset', type = str, default = 'mnist',
        help = 'the dataset, default = "mnist", default = ["mnist", "svhn", "fmnist"]')
    parser.add_argument('--subset', action = IntListParser, default = None,
        help = 'whether or not to load a subset of dataset, default = None')

    parser.add_argument('--attacker', action = DictParser, default = None,
        help = 'the information of the attacker, format: step_size=XX,threshold=XX,iter_num=XX,order=XX')

    parser.add_argument('--optim', type = str, default = 'adam',
        help = 'the type of the optimizer, default = "adam"')
    parser.add_argument('--lr', type = float, default = 1e-4,
        help = 'the learning rate, default = 1e-4')

    parser.add_argument('--gpu', type = str, default = "0",
        help = 'choose which gpu to use, default = "0"')
    parser.add_argument('--out_folder', type = str, default = None,
        help = 'the output folder')
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

    if args.out_folder is None:
        raise ValueError('you should specify the output folder')
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    model = MLP(in_dim = args.in_dim, hidden_dims = args.hidden_dims, out_dim = args.out_dim, nonlinearity = args.nonlinearity)
    criterion = nn.CrossEntropyLoss()
    model = model.cuda(device) if use_gpu else model
    criterion = criterion.cuda(device) if use_gpu else criterion

    if args.optim.lower() in ['sgd',]:
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 1e-4)
    elif args.optim.lower() in ['adam',]:
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas = (0.9, 0.99), weight_decay = 1e-4)
    else:
        raise ValueError('Unrecognized Optimizer: %s' % args.optim.lower())

    attacker = PGM(**args.attacker) if args.attacker != None else None

    if args.dataset.lower() in ['mnist',]:
        train_loader = load_mnist(batch_size = args.batch_size, dset = 'train', subset = args.subset)
        test_loader = load_mnist(batch_size = args.batch_size, dset = 'test', subset = args.subset)
        train_num_per_class = 5000
        test_num_per_class = 1000
    elif args.dataset.lower() in ['svhn',]:
        train_loader = load_svhn(batch_size = args.batch_size, dset = 'train', subset = args.subset)
        test_loader = load_svhn(batch_size = args.batch_size, dset = 'test', subset = args.subset)
        train_num_per_class = 7325
        test_num_per_class = 2603
    elif args.dataset.lower() in ['fmnist', 'fashionmnist']:
        train_loader = load_fmnist(batch_size = args.batch_size, dset = 'train', subset = args.subset)
        test_loader = load_fmnist(batch_size = args.batch_size, dset = 'test', subset = args.subset)
        train_num_per_class = 6000
        test_num_per_class = 1000
    else:
        raise ValueError('Unrecognized dataset %s'%args.dataset)

    train_batch_num = train_num_per_class * len(args.subset) // args.batch_size if args.subset != None else train_num_per_class * 10 // args.batch_size
    test_batch_num = test_num_per_class * len(args.subset) // args.batch_size if args.subset != None else test_num_per_class * 10 // args.batch_size

    setup_config = {kwarg: value for kwarg, value in args._get_kwargs()}

    acc_calculator = AverageCalculator()
    loss_calculator = AverageCalculator()
    setup_config['results'] = {'train_acc': {}, 'train_loss': {}, 'test_acc': {}, 'test_loss': {}}

    for epoch_idx in range(args.epochs):

        acc_calculator.reset()
        loss_calculator.reset()
        model.train()

        for idx in range(train_batch_num):

            data_batch, label_batch = next(train_loader)

            data_batch = data_batch.cuda(device) if use_gpu else data_batch
            label_batch = label_batch.cuda(device) if use_gpu else label_batch

            if attacker is not None:
                data_batch = attacker.attack(model, optimizer, data_batch, label_batch)

            logits = model(data_batch)
            loss = criterion(logits, label_batch)
            acc = accuracy(logits.data, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_calculator.update(acc.item(), args.batch_size)
            loss_calculator.update(loss.item(), args.batch_size)

        print('epoch %d: accuracy = %.2f%%' % (epoch_idx, acc_calculator.average * 100.))
        print('epoch %d: loss = %.4f' % (epoch_idx, loss_calculator.average))
        setup_config['train_acc'] = acc_calculator.average
        setup_config['train_loss'] = loss_calculator.average

        acc_calculator.reset()
        loss_calculator.reset()
        model.eval()

        for idx in range(test_batch_num):

            data_batch, label_batch = next(test_loader)

            data_batch = data_batch.cuda(device) if use_gpu else data_batch
            label_batch = label_batch.cuda(device) if use_gpu else label_batch

            logits = model(data_batch)
            loss = criterion(logits, label_batch)
            acc = accuracy(logits.data, label_batch)

            acc_calculator.update(acc.item(), args.batch_size)
            loss_calculator.update(loss.item(), args.batch_size)

        print('epoch %d: accuracy = %.2f%%' % (epoch_idx, acc_calculator.average * 100.))
        print('epoch %d: loss = %.4f' % (epoch_idx, loss_calculator.average))
        setup_config['test_acc'] = acc_calculator.average
        setup_config['test_loss'] = loss_calculator.average

    pickle.dump(setup_config, open(os.path.join(args.out_folder, '%s.pkl' % args.model_name), 'wb'))
    torch.save(model.state_dict(), os.path.join(args.out_folder, '%s.ckpt' % args.model_name))
