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

from util.param_parser import DictParser, ListParser, BooleanParser
from plot.color import get_color

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--bound_pkl', action = ListParser, default = None,
        help = 'the files containing the bound information')
    parser.add_argument('--labels', action = ListParser, default = None,
        help = 'the list of labels attached to each file')

    parser.add_argument('--bins', action = DictParser,
        default = {'min': 0., 'max': 0.1, 'bins': 100},
        help = 'bins of the histogram, default is min=0.,max=0.1,bins=100')
    parser.add_argument('--batch_idx', type = int, default = 0,
        help = 'the index of the batch, default = 0')
    parser.add_argument('--instance_idx', type = int, default = 0,
        help = 'the index of the instance, default = 0')

    parser.add_argument('--out_file', type = str, default = None,
        help = 'the output file')

    args = parser.parse_args()

    out_dir = os.path.dirname(args.out_file)
    if out_dir != '' and os.path.exists(args.out_dir) == False:
        os.makedirs(out_dir)

    bins = np.linspace(args.bins['min'], args.bins['max'], int(args.bins['bins']))

    for idx, (bound, label) in enumerate(zip(args.bound_pkl, args.labels)):

        bound_info = pickle.load(open(bound, 'rb'))['results']

        bound_selected = bound_info[args.batch_idx]['eps'][args.instance_idx].reshape(-1)

        plt.hist(bound_selected, bins, alpha = 0.5, color = get_color(idx), label = label)

    plt.legend(loc = 'upper right')
    plt.xlabel('bound')
    plt.ylabel('pixels')
    plt.savefig(args.out_file, dpi = 500, bbox = 'tight')

