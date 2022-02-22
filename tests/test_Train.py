import sys
sys.path.insert(1, '../lib')
import unittest
from Train import Train
from Inputs import Read_Input
import os

class TestModels(unittest.TestCase):

    path_data = '../data'
    input_data = Read_Input('test_inputs.yaml')
    Train(path_data, input_data)

    def test_check_runs_not_empty(self):

        runs_contents = os.listdir('runs')
        number_of_files = len(runs_contents)
        self.assertTrue(number_of_files == 1)

    def test_check_models_not_empty(self):

        models_contents = os.listdir('models')
        number_of_files = len(models_contents)
        self.assertTrue(number_of_files == 2)

    def test_check_epoch_match(self):
        pass

    def test_check_correct_model_used(self):
        pass



if __name__ == '__main__':
    unittest.main()
