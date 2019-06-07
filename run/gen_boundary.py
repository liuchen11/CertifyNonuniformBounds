import os
import sys
sys.path.insert(0, './')

import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn

from util.models import MLP
from util.device_parser import config_visible_gpu
from util.param_parser import DictParser, ListParser, IntListParser

def find_nearest_index(v_list, value, tolerance = 1e-4):
    '''
    >>> return the closest value in a list given a tolerance
    '''
    r_list = [np.abs(v - value) for v in v_list]
    index = np.argmin(r_list)
    if r_list[index] > tolerance:
        raise ValueError('There is no value close enough.')
    return index

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--min_x', type = float, default = -1.,
        help = 'the minimum value of x, default = -1.')
    parser.add_argument('--max_x', type = float, default = +1.,
        help = 'the maximum value of x, default = +1.')
    parser.add_argument('--num_x', type = int, default = 1000,
        help = 'the number of samples in x axis, default = 1000')
    parser.add_argument('--min_y', type = float, default = -1.,
        help = 'the minimum value of y, default = -1.')
    parser.add_argument('--max_y', type = float, default = +1.,
        help = 'the maximum value of y, default = +1.')
    parser.add_argument('--num_y', type = int, default = 1000,
        help = 'the number of samples in y axis, default = 1000')

    parser.add_argument('--model2load', type = str, default = None,
        help = 'the model to be loaded')
    parser.add_argument('--in_dim', type = int, default = 2,
        help = 'the number of input dimensions, default = 2')
    parser.add_argument('--hidden_dims', action = IntListParser, default = [],
        help = 'the number of neurons in hidden layers, default = []')
    parser.add_argument('--out_dim', type = int, default = 10,
        help = 'the number of classes, default = 10')
    parser.add_argument('--nonlinearity', type = str, default = 'relu',
        help = 'the non-linearity of the neural network, default = "relu"')

    parser.add_argument('--batch_size', type = int, default = 500,
        help = 'the batch size, default = 500')
    parser.add_argument('--gpu', type = str, default = '0',
        help = 'choose which gpu to use, default = "0"')
    parser.add_argument('--out_file', type = str, default = None,
        help = 'the output file')

    args = parser.parse_args()
    config_visible_gpu(args.gpu)
    if args.gpu not in ['cpu',] and torch.cuda.is_available():
        use_gpu = True
        device = torch.device('cuda:0')
    else:
        use_gpu = False
        device = torch.device('cpu')

    out_dir = os.path.dirname(args.out_file)
    if out_dir != '' and os.path.exists(out_dir) == False:
        os.makedirs(out_dir)

    # Model configuration
    model = MLP(in_dim = args.in_dim, hidden_dims = args.hidden_dims, out_dim = args.out_dim, nonlinearity = args.nonlinearity)
    model = model.cuda(device) if use_gpu else model
    ckpt = torch.load(args.model2load)
    model.load_state_dict(ckpt)
    model.eval()

    # Build scanning data
    nx = np.linspace(args.min_x, args.max_x, args.num_x)
    ny = np.linspace(args.min_y, args.max_y, args.num_y)
    x_list, y_list = np.meshgrid(nx, ny)
    x_list = x_list.reshape([-1, 1])
    y_list = y_list.reshape([-1, 1])

    xy_list = np.concatenate([x_list, y_list], axis = 1)
    batch_num = (xy_list.shape[0] - 1) // args.batch_size + 1

    # Obtain scanning results
    results = {}
    for batch_idx in range(batch_num):

        raw_data_batch = xy_list[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
        data_batch = torch.FloatTensor(raw_data_batch)
        data_batch = data_batch.cuda(device) if use_gpu else data_batch
        logits = model(data_batch)
        _, label_batch = torch.max(logits, dim = 1)

        data_batch = data_batch.data.cpu().numpy()
        label_batch = label_batch.data.cpu().numpy()

        for data, label in zip(raw_data_batch, label_batch):
            results[(data[0], data[1])] = label

    # Compression
    sx_list, sy_list = np.meshgrid(nx, ny, sparse = True)
    sx_list = list(sx_list.reshape(-1))
    sy_list = list(sy_list.reshape(-1))

    boundary_pts = {}
    shift_list = [(0, -1), (0, +1), (-1, 0), (+1, 0)]

    total_items = len(results.keys())
    for idx, (data, label) in enumerate(results.items()):

        sys.stdout.write('Processing items %d / %d = %.2f%%\r' % (idx, total_items, float(idx) / float(total_items) * 100.))
        is_boundary = False

        x, y = data
        x_index = find_nearest_index(sx_list, x)
        y_index = find_nearest_index(sy_list, y)

        for shift_x, shift_y in shift_list:

            if shift_x + x_index < 0 or shift_x + x_index >= len(sx_list) \
                 or shift_y + y_index < 0 or shift_y + y_index >= len(sy_list):
                continue

            neighbor_x = sx_list[x_index + shift_x]
            neighbor_y = sy_list[y_index + shift_y]
            neighbor = xy_list[shift_x + x_index + (shift_y + y_index) * len(sy_list)]

            assert np.abs(neighbor_x - neighbor[0]) < 1e-4
            assert np.abs(neighbor_y - neighbor[1]) < 1e-4

            neighbor_label = results[(neighbor[0], neighbor[1])]
            if neighbor_label != label:
                is_boundary = True
                break

        if is_boundary == True:
            boundary_pts[(x, y)] = label

    tosave = {'config': {kwarg: value for kwarg, value in args._get_kwargs()}, 'results': boundary_pts}
    pickle.dump(tosave, open(args.out_file, 'wb'))

