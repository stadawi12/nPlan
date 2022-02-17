import unittest
import numpy as np

class TestData(unittest.TestCase):

    def test_train_length(self):
        # test to see if train_feats is same length as train_labels
        data_feats  = np.load('../data/train_feats.npy')
        data_labels = np.load('../data/train_labels.npy')

        self.assertEqual(data_feats.shape[0], data_labels.shape[0])

    def test_test_length(self):
        # test to see if test_feats is same length as test_labels
        data_feats  = np.load('../data/test_feats.npy')
        data_labels = np.load('../data/test_labels.npy')

        self.assertEqual(data_feats.shape[0], data_labels.shape[0])

    def test_valid_length(self):
        # test to see if valid_feats is same length as valid_labels
        data_feats  = np.load('../data/valid_feats.npy')
        data_labels = np.load('../data/valid_labels.npy')

        self.assertEqual(data_feats.shape[0], data_labels.shape[0])


if __name__ == '__main__':
    unittest.main()
