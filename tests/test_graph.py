import sys
sys.path.insert(1, "../lib")
import unittest
from graph import Graphs

class TestModels(unittest.TestCase):

    
    def test_no_missing_links_train(self):
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

    def test_shift_train(self):
        pass




if __name__ == '__main__':
    unittest.main()
