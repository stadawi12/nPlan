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

        # load inputs file for training
        input_data = Read_Input('inputs.yaml')
        
        # Grab some inputs
        use_seed: bool   = input_data["use_seed"]
        seed: int        = input_data["seed"]
        record_run: bool = input_data["record_run"]
        save_model: bool = input_data["save_model"]

        path_data = 'data'
        input_data = Read_Input('inputs.yaml')

        if record_run == False:
            print("WARNING: run will NOT be recorded!")

        if save_model == False:
            print("WARNING: model will NOT be saved!")

        
        if use_seed:
            torch.manual_seed(seed)
            print("Using seed: ", seed)

        Train(path_data, input_data)

    if args.action == 'test':
        # import Test module
        from lib.Test import Test
        
        # specify arguments of Test function
        path_data = 'data'
        path_model = 'models'
        path_inputs = 'Test_inputs.yaml'

        input_data = Read_Input(path_inputs)
        
        Test(path_data, path_model, input_data)
