import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import argparse
from DataManipulator import DataManipulator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate aligned objs from the original ones.')
    parser.add_argument('-d', '--data-path', dest='path', type=str, default='../../3Rscan/data/', help='3Rscan data path')
    args = parser.parse_args()
    print(args)

    data_manip = DataManipulator(args.path)
    data_manip.create_located_obj()
