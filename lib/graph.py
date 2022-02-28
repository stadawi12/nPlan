import numpy as np
import json
import os
import torch

class Graphs():

    """Class to work with graph data, can be used for extracting
    edge_indexes for a specific graph"""

    def __init__(self, path_data: str, dataset: str):
        """Parameters
        ----------
        path_data : str
            this is the path to the data directory where all graph data is
            stored, for example, if you are running this function from the
            main directory you would specify path_data = 'data', the default
            is '.' (empty) meaning we are already in the data directory
        dataset : str
            this option allows you to choose 'train', 'test' or 'valid'
            datasets
        """

        # set path to data directory and assert that it exists
        self.path_data = path_data
        MSG1 = f"Path {self.path_data} does not exist"
        assert os.path.exists(self.path_data), MSG1

        # make sure dataset is set to one of the valid options
        self.dataset = dataset
        MSG21 = f"dataset value '{self.dataset}' is not available. "
        MSG22 = "Use one from 'train', 'test' or 'valid'"
        MSG2 = MSG21 + MSG22
        assert self.dataset in ['train', 'test', 'valid'], MSG2

        # create filename of the grah_id.npy file
        self.filename_ids = self.dataset + '_graph_id.npy'

        # create path to graph_ids.npy file and assert it exists
        self.path_ids = os.path.join(self.path_data, self.filename_ids)
        MSG3 = f"Path {self.path_ids} does not exist"
        assert os.path.exists(self.path_ids), MSG3

        # create filename of the graph.json file
        self.filename_json = self.dataset + '_graph.json'

        # create path to graph.json file and assert it exists
        self.path_json = os.path.join(self.path_data, self.filename_json)
        MSG4 = f"Path {self.path_json} does not exist"
        assert os.path.exists(self.path_json), MSG4

        # create path to feats.npy file and assert it exists
        self.path_feats = os.path.join(self.path_data,
                'features', self.dataset)
        MSG5 = f"Path {self.path_feats} does not exist"
        assert os.path.exists(self.path_feats), MSG5

        # create filename of the labels.npy file
        self.filename_labels = self.dataset + '_labels.npy'
        # create path to labels.npy file and assert it exists
        self.path_labels = os.path.join(self.path_data,
                self.filename_labels)
        MSG6 = f"Path {self.path_labels} does not exist"
        assert os.path.exists(self.path_labels), MSG6

    def get_shift(self, i):
        # load graph_id.npy file as a numpy array
        # this file contains the graph ids that each node belongs to
        # series of [1,1,1,...,2,2,2,...,3,3,.....]
        graph_ids = np.load(self.path_ids)
        MSG11 = f"graph_ids are not sorted"
        assert list(graph_ids) == sorted(list(graph_ids)), MSG11

        # calculate how many nodes in each graph using dictionary
        # counting, counts = {graph_id: count}
        counts = {x: 0 for x in sorted(set(graph_ids))}
        for el in graph_ids:
            counts[el] += 1

        keys = list(counts.keys())

        chosen_graph_id = keys[i]

        # get the smallest graph id in our dataset, if list of graph ids
        # is like [1,2,3,4,...], smallest_graph_id would be 1
        smallest_graph_id = keys[0]

        # this specifies the shift to adjacency matrix, if
        # chosen_graph_id is the same as smallest_graph_id then there is
        # no shift.         
        shift = 0

        if chosen_graph_id > smallest_graph_id:
            # get ids of all graphs which are below the one we are
            # interested in
            ids_above = []
            for key in keys:
                if key < chosen_graph_id:
                    ids_above.append(key)
            # shift is equal to the sum of sizes of adjacency matrices
            # that are above it
            shift = sum([counts[ID] for ID in ids_above])

        return shift

    # this specifies the indexing of our object
    def get_edges(self, i):
        """ using the index we will return an features and edge_index
        array for a single graph indexed by i in our dataset """

        # load graph_id.npy file as a numpy array
        # this file contains the graph ids that each node belongs to
        # series of [1,1,1,...,2,2,2,...,3,3,.....]
        graph_ids = np.load(self.path_ids)

        MSG11 = f"graph_ids are not sorted"
        assert list(graph_ids) == sorted(list(graph_ids)), MSG11

        keys = sorted(set(graph_ids))
        
        # extract chosen_graph_id, given index = 0, which graph id are we
        # asking for, that's what chosen_graph_id is
        chosen_graph_id = keys[i]

        # generate an empty list for holding edge_indexes 
        edge_index = []

        shift = self.get_shift(i)

        # open the json file with all edges between nodes for our
        # dataset
        with open(self.path_json) as jsonFile:
            jsonObject = json.load(jsonFile)
            
            # get the dictionary of links (source/target pairs)
            links = jsonObject['links']

            # go through the list of all links (edges)
            for link in links:

                # get the source and target of a single link
                source = link['source']
                target = link['target']

                # check that the graph_id of source and target are the
                # same as the graph_id that we are interested in 
                if graph_ids[source] == graph_ids[target] == chosen_graph_id:

                    # update the adjacency to include the link
                    # information
                    edge_index.append([source - shift, target - shift])

            jsonFile.close()

        edge_index = torch.tensor(edge_index, dtype=torch.long)

        return edge_index.t().contiguous()

    def get_labels(self, i):

        # load graph_id.npy file as a numpy array
        # this file contains the graph ids that each node belongs to
        # series of [1,1,1,...,2,2,2,...,3,3,.....]
        graph_ids = np.load(self.path_ids)
        MSG11 = f"graph_ids are not sorted"
        assert list(graph_ids) == sorted(list(graph_ids)), MSG11

        # calculate how many nodes in each graph using dictionary
        # counting, counts = {graph_id: count}
        counts = {x: 0 for x in sorted(set(graph_ids))}
        for el in graph_ids:
            counts[el] += 1

        keys = list(counts.keys())

        chosen_graph_id = keys[i]

        # load labels (num_nodes, 121)
        labels = torch.from_numpy(np.load(self.path_labels))

        shift = self.get_shift(i)

        return labels[shift : shift + counts[chosen_graph_id]]

    def get_features(self, i):

        filename = 'id_' + str(i) + '.pt'
        print(filename)

        path_features = os.path.join(self.path_feats, filename)

        features = torch.load(path_features)

        return features


    def __len__(self):
        # return the number of graphs found in dataset
        graph_ids = np.load(self.path_ids)

        ids = set(graph_ids)

        return len(ids)


if __name__ == '__main__':
    with open('../data/train_graph.json') as f:
        graph = json.load(f)
        links = graph['links']
        print(len(links))

    g = Graphs('../data', 'valid')
    # count_edges = 0
    # for i in range(len(g)):
    #     edges = g.get_edges(i)
    #     count_edges += edges.shape[1]
    # print(count_edges)
    idx = 1
    features = g.get_labels(idx)
    labels = g.get_labels(idx)
    print(features.shape)
    print(labels.shape)
