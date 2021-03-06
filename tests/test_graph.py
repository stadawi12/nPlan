import sys
sys.path.insert(1, "../lib")
import unittest
from graph import Graphs

class TestModels(unittest.TestCase):

    
    def test_no_missing_links_test(self):
        import json

        g = Graphs('../data', 'test')

        # count of links
        links_count = 0
        for idx in range(len(g)):
            edges = g.get_edges(idx)
            links_count += edges.shape[1]

        with open('../data/test_graph.json') as f:
            graphs = json.load(f)
            links  = graphs['links']
            
        self.assertEqual(links_count, len(links))

    def test_no_missing_links_valid(self):
        import json

        g = Graphs('../data', 'valid')

        # count of links
        links_count = 0
        for idx in range(len(g)):
            edges = g.get_edges(idx)
            links_count += edges.shape[1]

        with open('../data/valid_graph.json') as f:
            graphs = json.load(f)
            links  = graphs['links']
            
        self.assertEqual(links_count, len(links))

    def test_equal_feature_label_len_train0(self):

        g = Graphs('../data', 'train')
        feats  = g.get_features(0)
        labels = g.get_labels(0)
        self.assertEqual(feats.shape[0], labels.shape[0])

    def test_equal_feature_label_len_train7(self):

        g = Graphs('../data', 'train')
        feats  = g.get_features(7)
        labels = g.get_labels(7)
        self.assertEqual(feats.shape[0], labels.shape[0])

    def test_equal_feature_label_len_valid0(self):

        g = Graphs('../data', 'valid')
        feats  = g.get_features(0)
        labels = g.get_labels(0)
        self.assertEqual(feats.shape[0], labels.shape[0])

    def test_equal_feature_label_len_valid1(self):

        g = Graphs('../data', 'valid')
        feats  = g.get_features(1)
        labels = g.get_labels(1)
        self.assertEqual(feats.shape[0], labels.shape[0])

    def test_equal_feature_label_len_test0(self):

        g = Graphs('../data', 'test')
        feats  = g.get_features(0)
        labels = g.get_labels(0)
        self.assertEqual(feats.shape[0], labels.shape[0])

    def test_equal_feature_label_len_test1(self):

        g = Graphs('../data', 'test')
        feats  = g.get_features(1)
        labels = g.get_labels(1)
        self.assertEqual(feats.shape[0], labels.shape[0])

    def test_len_of_graphs_train(self):

        g = Graphs('../data', 'train')

        l = len(g)

        self.assertEqual(l, 20)
        
    def test_len_of_graphs_test(self):

        g = Graphs('../data', 'test')

        l = len(g)

        self.assertEqual(l, 2)

    def test_len_of_graphs_valid(self):

        g = Graphs('../data', 'valid')

        l = len(g)

        self.assertEqual(l, 2)

    def test_shift_train2(self):

        g = Graphs('../data', 'train')

        shift = g.get_shift(2)

        self.assertEqual(shift, 3144)

    def test_shift_train0(self):

        g = Graphs('../data', 'train')

        shift = g.get_shift(0)

        self.assertEqual(shift, 0)

    def test_num_nodes_train(self):

        g = Graphs('../data', 'train')

        num_nodes = g.num_nodes(0)
        
        self.assertEqual(num_nodes, 1767)

        num_nodes = g.num_nodes(6)
        
        self.assertEqual(num_nodes, 1823)

    def test_num_nodes_valid(self):

        g = Graphs('../data', 'valid')

        num_nodes = g.num_nodes(0)
        
        self.assertEqual(num_nodes, 3230)

        num_nodes = g.num_nodes(1)
        
        self.assertEqual(num_nodes, 3284)

if __name__ == '__main__':
    unittest.main()
