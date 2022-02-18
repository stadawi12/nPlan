# - test_loader.py
import sys
sys.path.insert(1, '../lib') # path where loader is found
import unittest
from loader import dataset
from torch.utils.data import DataLoader
import numpy as np
import torch

class TestLoader(unittest.TestCase):
    """Here we will focus on testing the loader module designed for
    loading data for training, testing and validating"""

    PATH_DATA = '../data'

    def test_initialise_dataset_object(self):
        # test to see if we can initialise the dataset object correctly
        # and see if the attributes are correctly assigned
        data_train = dataset(self.PATH_DATA, 'train')
        data_test = dataset(self.PATH_DATA, 'test')
        data_valid = dataset(self.PATH_DATA, 'valid')

        self.assertEqual(data_train.path_data, '../data')
        self.assertEqual(data_train.data_for, 'train')
        self.assertEqual(data_test.data_for, 'test')
        self.assertEqual(data_valid.data_for, 'valid')

    def test_test_data(self):
        # test to see if we can also load test data correctly
        data_test = dataset(self.PATH_DATA, 'test')

        self.assertEqual(data_test.path_data, '../data')
        self.assertEqual(data_test.data_for, 'test')

    def test_valid_data(self):
        # test to see if we can also load valid data correctly
        data_valid = dataset(self.PATH_DATA, 'valid')

        self.assertEqual(data_valid.path_data, '../data')
        self.assertEqual(data_valid.data_for, 'valid')

    def test_len_dataset(self):
        # test to check if len() method works as it should
        data_train = dataset(self.PATH_DATA, 'train')
        data_test = dataset(self.PATH_DATA, 'test')
        data_valid = dataset(self.PATH_DATA, 'valid')

        self.assertEqual(len(data_train), 44906)
        self.assertEqual(len(data_test), 5524)
        self.assertEqual(len(data_valid), 6514)

    def test_indexing_dataset(self):
        # test to see if indexing works as it should and check against
        # loading data with numpy
        
        # First, load data using my loader module
        data_loader = dataset(self.PATH_DATA, 'train')
        # get the first 10 rows of data (both features and labels)
        loader_feats, loader_labels = data_loader[0:10]
        # load the same data using numpy
        numpy_feats = np.load('../data/train_feats.npy')
        numpy_labels = np.load('../data/train_labels.npy')

        tensor_feats  = torch.from_numpy(numpy_feats[0:10]).float()
        tensor_labels = torch.from_numpy(numpy_labels[0:10]).float()

        # check to see if the first 10 rows agree in values
        self.assertTrue(torch.equal(loader_feats, tensor_feats))
        self.assertTrue(torch.equal(loader_labels, tensor_labels))

    def test_dataloader_type(self):
        # test to make sure that the data type of our data point are
        # floats 

        # First, load data using my loader module
        data_train = dataset(self.PATH_DATA, 'train')
        # use DataLoader to manage loading training data
        loader_train = DataLoader(data_train, shuffle=False,
                batch_size=1)
        train_f, train_l = next(iter(loader_train))

        self.assertTrue(train_f.dtype == torch.Tensor(1).dtype)
        self.assertTrue(train_l.dtype == torch.Tensor(1).dtype)



if __name__ == "__main__":
    unittest.main()
