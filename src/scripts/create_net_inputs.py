import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import argparse
from DataManipulator import DataManipulator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate classifier network inputs by sampling data points from the original pointclouds.')
    parser.add_argument('-s', '--sample-size', dest='ssize', type=int, required=True, help='sample generation dimension')
    parser.add_argument('-d', '--data-path', dest='path', type=str, default='../../3Rscan/data/', help='3Rscan data path')
    args = parser.parse_args()
    print(args)

    data_manip = DataManipulator(args.path)
    data_manip.create_sampled_inputs(args.ssize)
