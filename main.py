import sys 
sys.path.insert(1, 'lib')
sys.path.insert(1, 'lib/models')

import argparse
from lib.Train import Train
from lib.Inputs import Read_Input, Create_Parser
import torch

if __name__ == '__main__':

    args = Create_Parser()

    if args.action == 'train':

        path_data = 'data'
        input_data = Read_Input('inputs.yaml')

        torch.manual_seed(0)

        Train(path_data, input_data)
