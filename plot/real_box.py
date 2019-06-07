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
import matplotlib.gridspec as gridspec

from util.param_parser import ListParser, FloatListParser
from plot.imshow import imshow

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--bound_pkl', action = ListParser, default = None,
        help = 'the files containing the bound information')
    parser.add_argument('--labels', action = ListParser, default = None,
        help = 'the list of labels')

    parser.add_argument('--batch_size', type = int, default = 5,
        help = 'the size of batch, default = 5')
    parser.add_argument('--batch_idx', type = int, default = 0,
        help = 'specify the batch idx to visualize, default = 0')
    parser.add_argument('--rescale', action = FloatListParser, default = [0., 0.2],
        help = 'rescaling factor (bias, scale), default = None')
    parser.add_argument('--out_folder', type = str, default = None,
        help = 'the folder to save individual images')

    args = parser.parse_args()

    if os.path.exists(args.out_folder) == False:
        os.makedirs(args.out_folder)

    cols = args.batch_size
    rows = len(args.bound_pkl) * 2

    plt.figure(figsize = (rows, cols))
    gs = gridspec.GridSpec(rows, cols)

    # Plot joint figure
    print('Generating joint figure of bounding box ...')
    for file_idx, pkl_file in enumerate(args.bound_pkl):

        file_data = pickle.load(open(pkl_file, 'rb'))
        results = file_data['results']

        picked_batch = results[args.batch_idx]
        assert np.sum(picked_batch['result_mask']) >= args.batch_size, 'not enough valid data'

        plot_idx = 0
        for image_idx, mask in enumerate(picked_batch['result_mask']):
            if mask < 0.5:
                continue

            image_data = (picked_batch['data_batch'][image_idx].reshape(28, 28, 1) + 1.) / 2.
            box_data = picked_batch['eps'][image_idx].reshape(28, 28, 1)

            image_plot_idx = file_idx * 2 * cols + plot_idx
            bound_plot_idx = (file_idx * 2 + 1) * cols + plot_idx

            plt.subplot(gs[image_plot_idx])
            imshow(image = image_data)
            plt.subplot(gs[bound_plot_idx])
            imshow(image = box_data, rescale = args.rescale, complementary = True)

            plot_idx += 1
            if plot_idx >= args.batch_size:
                break

    plt.savefig(os.path.join(args.out_folder, 'box.pdf'), dpi = 500, bbox_inches = 'tight')
    plt.clf()
    print('Completed!')

    # Plot individual figure
    print('Generating individual figures of bounding box ...')
    for file_idx, (pkl_file, label_file) in enumerate(zip(args.bound_pkl, args.labels)):

        file_data = pickle.load(open(pkl_file, 'rb'))
        results = file_data['results']

        picked_batch = results[args.batch_idx]

        plot_idx = 0
        for image_idx, mask in enumerate(picked_batch['result_mask']):
            if mask < 0.5:
                continue

            image_data = (picked_batch['data_batch'][image_idx].reshape(28, 28, 1) + 1.) / 2.
            box_data = picked_batch['eps'][image_idx].reshape(28, 28, 1)

            imshow(image = image_data)
            plt.savefig(os.path.join(args.out_folder, '%d_%s_image.pdf' % (plot_idx, label_file)), dpi = 500, bbox_inches = 'tight')
            plt.clf()
            imshow(image = box_data, rescale = args.rescale, complementary = True)
            plt.savefig(os.path.join(args.out_folder, '%d_%s_box.pdf' % (plot_idx, label_file)), dpi = 500, bbox_inches = 'tight')
            plt.clf()

            plot_idx += 1
            if plot_idx >= args.batch_size:
                break
    print('Completed!')
