import os
import sys
sys.path.insert(0, './')

import pickle
import argparse
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dim', type = int, default = 2,
        help = 'the number of dimensions, default = 2')
    parser.add_argument('--pts', type = int, default = 10000,
        help = 'the number of points, default = 10000')
    parser.add_argument('--classes', type = int, default = 10,
        help = 'the number of classes, default = 10')

    parser.add_argument('--out_file', type = str, default = None,
        help = 'the output file')

    args = parser.parse_args()

    if args.out_file is None:
        raise ValueError('the output file need to be specified.')
    if not os.path.exists(os.path.dirname(args.out_file)):
        os.makedirs(os.path.dirname(args.out_file))

    # construct base points
    base_pts = [np.random.uniform(low = -1., high = 1., size = [args.dim,]) for _ in range(args.classes)]
    base_pts = np.array(base_pts)
    print('Base points constructed!')

    # construct data points
    data_set = []
    label_set = []
    for idx in range(args.pts):
        sys.stdout.write('%d / %d loaded\r' % (idx + 1, args.pts))
        data_pt = np.random.uniform(low = -1., high = 1., size = [args.dim,])
        distance_list = [(idx, np.linalg.norm(base_pt - data_pt) ** 2) for idx, base_pt in enumerate(base_pts)]
        distance_list = sorted(distance_list, key = lambda x: x[1])
        label_pt = distance_list[0][0]
        data_set.append(data_pt)
        label_set.append(label_pt)

    data_set = np.array(data_set)
    label_set = np.array(label_set, dtype = int)
    print('Data points constructed!')

    # calculate boundary
    boundary_pts = []
    for base_idx1 in range(args.classes):

        for base_idx2 in range(base_idx1 + 1, args.classes):

            pt1 = base_pts[base_idx1]
            pt2 = base_pts[base_idx2]

            mid = (pt1 + pt2) / 2.
            arr = pt2 - pt1
            arr = np.array([arr[1], - arr[0]]) / np.linalg.norm(arr)

            min_x = (- 1. - mid[0]) / arr[0]
            max_x = (1. - mid[0]) / arr[0]
            min_y = (- 1. - mid[1]) / arr[1]
            max_y = (1. - mid[1]) / arr[1]

            _, min_idx, max_idx, _ = list(sorted([min_x, min_y, max_x, max_y]))

            for idx in np.arange(min_idx, max_idx, 0.005):
                pt = mid + idx * arr
                boundary = True
                dis = np.linalg.norm(pt - pt1)

                for base_idx in range(args.classes):
                    if base_idx in [base_idx1, base_idx2]:
                        continue

                    dis_ = np.linalg.norm(pt - base_pts[base_idx])
                    if dis_ < dis:
                        boundary = False
                        break

                if boundary == True:
                    boundary_pts.append((pt[0], pt[1]))
    print('Boundary points constructed!')

    pickle.dump({'data': data_set, 'label': label_set, 'base_points': base_pts,
        'classes': args.classes, 'boundary': boundary_pts}, open(args.out_file, 'wb'))
    print('Information dumpped in file %s' % args.out_file)

