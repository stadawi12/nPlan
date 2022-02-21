import sys 
sys.path.insert(1, 'lib')
sys.path.insert(1, 'lib/models')

import argparse
from lib.Train import Train
from lib.Inputs import Read_Input, Create_Parser
import torch

if __name__ == '__main__':

    args = Create_Parser()
    input_data = Read_Input('inputs.yaml')

    use_seed: bool = input_data["use_seed"]
    seed: int      = input_data["seed"]

    if args.action == 'train':

        path_data = 'data'
        input_data = Read_Input('inputs.yaml')

        
        if use_seed:
            torch.manual_seed(seed)
            print("Using seed: ", seed)

        Train(path_data, input_data)
