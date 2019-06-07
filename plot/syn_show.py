import os
import sys
sys.path.insert(0, './')

import pickle
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

from util.param_parser import DictParser, ListParser, IntListParser
from plot.color import get_color

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_pkl', type = str, default = None,
        help = 'the data file to be loaded.')
    parser.add_argument('--boundary_pkl', type = str, default = None,
        help = 'the file containing models\'s boundary information')
    parser.add_argument('--bound_pkl', action = ListParser, default = None,
        help = 'the bound file(s) to be loaded.')
    parser.add_argument('--out_file', type = str, default = None,
        help = 'the output file.')

    parser.add_argument('--batch_num', type = int, default = None,
        help = 'the number of batches selected to plot, default = None, meaning all.')
    parser.add_argument('--pts_per_class', type = int, default = None,
        help = 'the number of pts per class, default = None, meaning no limitations.')

    args = parser.parse_args()

    data_info = pickle.load(open(args.data_pkl, 'rb'))
    out_dir = os.path.dirname(args.out_file)
    if out_dir != '' and os.path.exists(out_dir) == False:
        os.makedirs(out_dir)

    base_pts = data_info['base_points']
    out_dim = data_info['classes']

    # plot the boundary
    boundary_info = pickle.load(open(args.boundary_pkl, 'rb'))
    boundary_pts = np.array(list(boundary_info['results'].keys()))
    plt.scatter(boundary_pts[:, 0], boundary_pts[:, 1], color = 'black', s = 2)
    print('Boundary pts plotted.')

    # Load the bound
    for idx, bound_file in enumerate(args.bound_pkl):
        file_color = get_color(idx + 1)
        bound_info = pickle.load(open(bound_file, 'rb'))
        counts = [0 for _ in range(out_dim)]

        batch_num = args.batch_num if args.batch_num is not None else len(bound_info['results'])
        for batch_idx in range(batch_num):
            data_batch = bound_info['results'][batch_idx]['data_batch']
            result_mask = bound_info['results'][batch_idx]['result_mask']
            eps_batch = bound_info['results'][batch_idx]['eps']

            for p_idx, (data, result, eps) in enumerate(zip(data_batch, result_mask, eps_batch)):

                if result == 0:
                    continue

                dis_list = [np.linalg.norm(data - base_pt) for base_pt in base_pts]
                predict = np.argmin(dis_list)
                if args.pts_per_class != None and counts[predict] >= args.pts_per_class:
                    continue
                counts[predict] += 1

                plt.scatter([data[0],], [data[1],], s = 3, color = 'b')
                plt.text(x = data[0] + 1e-2, y = data[1] + 1e-2, s = str(np.sum(counts)), color = 'b', fontsize = 5)
                plt.plot([data[0] - eps[0], data[0] - eps[0]], [data[1] - eps[1], data[1] + eps[1]], color = file_color)
                plt.plot([data[0] + eps[0], data[0] + eps[0]], [data[1] - eps[1], data[1] + eps[1]], color = file_color)
                plt.plot([data[0] - eps[0], data[0] + eps[0]], [data[1] - eps[1], data[1] - eps[1]], color = file_color)
                plt.plot([data[0] - eps[0], data[0] + eps[0]], [data[1] + eps[1], data[1] + eps[1]], color = file_color)
        print('counts', counts)

    plt.xticks([-1., 0., 1.])
    plt.yticks([-1., 0., 1.])
    plt.xlim(-1., 1.)
    plt.ylim(-1., 1.)
    plt.savefig(args.out_file, bbox = 'tight', dpi = 500)

