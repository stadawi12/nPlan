import sys 
sys.path.insert(1, 'lib')
sys.path.insert(1, 'lib/models')

import argparse
from lib.Train import Train
from lib.Inputs import Read_Input, Create_Parser
from lib.graph import Graphs
import torch

if __name__ == '__main__':

    args = Create_Parser()

    if args.action == 'classify':
        
        import math

        # load inputs file for training
        input_data = Read_Input('inputs.yaml')
        
        # Grab some inputs
        use_seed: bool   = input_data["use_seed"]
        seed: int        = input_data["seed"]
        record_run: bool = input_data["record_run"]
        save_model: bool = input_data["save_model"]
        dataset: str     = input_data["dataset"]
        graph_id: int    = input_data["graph_id"]
        test_split: float = input_data["test_split"]

        path_data = 'data'

        if record_run == False:
            print("WARNING: run will NOT be recorded!")

        if save_model == False:
            print("WARNING: model will NOT be saved!")

        
        if use_seed:
            torch.manual_seed(seed)
            print("Using seed: ", seed)

        data = Graphs('data', dataset)

        if graph_id != None:

            features = data.get_features(graph_id)
            labels = data.get_labels(graph_id)

            n_examples = len(features)
            n_test = math.ceil(test_split * n_examples)

            train_x = features[:-n_test].float()
            train_y = labels[:-n_test].float()
            test_x = features[-n_test:].float()
            test_y = labels[-n_test:].float()

            Train(train_x, train_y, test_x, test_y, input_data)

        else:

            for i in range(len(data)):

                features = data.get_features(i)
                labels = data.get_labels(i)

                n_examples = len(features)
                n_test = math.ceil(test_split * n_examples)

                train_x = features[:-n_test].float()
                train_y = labels[:-n_test].float()
                test_x = features[-n_test:].float()
                test_y = labels[-n_test:].float()

                Train(train_x, train_y, test_x, test_y, input_data)

    if args.action == 'embed':
        # import Test module
        from lib.Test import Test
        
        # specify arguments of Test function
        path_data = 'data'
        path_model = 'models'
        path_inputs = 'Test_inputs.yaml'

        input_data = Read_Input(path_inputs)
        
        Test(path_data, path_model, input_data)
