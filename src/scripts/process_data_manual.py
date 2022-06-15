import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import argparse
from DataManipulator import DataManipulator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='All in one 3Rscan data processor and generator.')
    parser.add_argument('-so', '--sample-size-objects', dest='ssize_obj', type=int, default=20000, help='objects sample generation dimension')
    parser.add_argument('-sm', '--sample-size-mask', dest='ssize_msk', type=int, default=-1, help='mask sample generation dimension')
    parser.add_argument('-e', '--enhance', dest='enhance', type=float, default=.6, help='bbox enhance value')
    parser.add_argument('-d', '--data-path', dest='path', type=str, default='../../3Rscan/data/', help='3Rscan data path')
    parser.add_argument('-c', '--cap-path', dest='c_path', type=str, default='../../manual-captions/', help='captions data path')
    parser.add_argument('-i', '--incremental', dest='incremental', type=bool, default=False, help='create only newly added samples')
    args = parser.parse_args()
    print(args)

    data_manip = DataManipulator(args.path)
    data_manip.create_manual_objs(args.c_path, args.enhance, args.incremental)
    data_manip.create_manual_plys(args.c_path, args.enhance, args.incremental)
    data_manip.create_sampled_single_objects(args.ssize_obj, args.c_path, args.incremental)
    if args.ssize_msk != -1:
        data_manip.create_manual_masks(args.ssize_msk, args.c_path, args.incremental)
