import argparse
from lib.Train import Train
from lib.Inputs import Read_Input, Create_Parser
import torch

if __name__ == '__main__':

    path_data = 'data'
    input_data = Read_Input('inputs.yaml')

    torch.manual_seed(0)

    Train(path_data, input_data)
