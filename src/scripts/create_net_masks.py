import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import argparse
from DataManipulator import DataManipulator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate classifier network inputs by sampling data points from the original pointclouds.')
    parser.add_argument('-s', '--sample-size', dest='ssize', type=int, default=75000, help='sample generation dimension')
    parser.add_argument('-d', '--data-path', dest='d_path', type=str, default='../../3Rscan/data/', help='3Rscan data path')
    parser.add_argument('-c', '--cap-path', dest='c_path', type=str, default='../../manual-captions/', help='captions data path')
    parser.add_argument('-i', '--incremental', dest='incremental', type=bool, default=False, help='create only newly added samples')
    args = parser.parse_args()
    print(args)

    data_manip = DataManipulator(args.d_path)
    data_manip.create_manual_masks(args.ssize, args.c_path, args.incremental)
