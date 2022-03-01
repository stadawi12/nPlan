import sys
sys.path.insert(1, '../lib')
import unittest
import numpy as np
from graph import Graphs
import torch


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

    def test_distinct_graphs_test0(self):

        g = Graphs('../data', 'test')

        num_nodes = g.num_nodes(0)

        edges = g.get_edges(0)

        self.assertEqual(torch.min(edges), 0)
        self.assertEqual(torch.max(edges), num_nodes-1)

    def test_distinct_graphs_test1(self):

        g = Graphs('../data', 'test')

        num_nodes = g.num_nodes(1)

        edges = g.get_edges(1)

        self.assertEqual(torch.min(edges), 0)
        self.assertEqual(torch.max(edges), num_nodes-1)

    def test_distinct_graphs_valid0(self):

        g = Graphs('../data', 'valid')

        num_nodes = g.num_nodes(0)

        edges = g.get_edges(0)

        self.assertEqual(torch.min(edges), 0)
        self.assertEqual(torch.max(edges), num_nodes-1)

    def test_distinct_graphs_valid1(self):

        g = Graphs('../data', 'valid')

        num_nodes = g.num_nodes(1)

        edges = g.get_edges(1)

        self.assertEqual(torch.min(edges), 0)
        self.assertEqual(torch.max(edges), num_nodes-1)


if __name__ == '__main__':
    unittest.main()
