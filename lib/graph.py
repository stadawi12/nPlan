import numpy as np
import json
import os
import torch

class Graphs():

    """Class to work with graph data, can be used for extracting
    edge_indexes for a specific graph, features, labels and other
    useful attributes"""

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
        MSG1  = f"Path {self.path_data} does not exist"
        assert os.path.exists(self.path_data), MSG1


        # make sure dataset is set to one of the valid options
        self.dataset = dataset
        MSG21 = f"dataset value '{self.dataset}' is not available. "
        MSG22 = "Use one from 'train', 'test' or 'valid'"
        MSG2  = MSG21 + MSG22
        assert self.dataset in ['train', 'test', 'valid'], MSG2


        # create filename of the grah_id.npy file
        self.filename_ids = self.dataset + '_graph_id.npy'
        # create path to graph_id.npy file and assert it exists
        self.path_ids = os.path.join(self.path_data, self.filename_ids)
        MSG3  = f"Path {self.path_ids} does not exist"
        assert os.path.exists(self.path_ids), MSG3


        # create filename of the graph.json file
        self.filename_json = self.dataset + '_graph.json'
        # create path to graph.json file and assert it exists
        self.path_json = os.path.join(self.path_data, self.filename_json)
        MSG4  = f"Path {self.path_json} does not exist"
        assert os.path.exists(self.path_json), MSG4


        # create path to feats.npy file and assert it exists
        self.path_feats = os.path.join(self.path_data,
                'features', self.dataset)
        MSG5  = f"Path {self.path_feats} does not exist"
        assert os.path.exists(self.path_feats), MSG5


        # create filename of the labels.npy file
        self.filename_labels = self.dataset + '_labels.npy'
        # create path to labels.npy file and assert it exists
        self.path_labels = os.path.join(self.path_data,
                self.filename_labels)
        MSG6  = f"Path {self.path_labels} does not exist"
        assert os.path.exists(self.path_labels), MSG6

        # load all ids of dataset and check if they are sorted
        self.all_ids = np.load(self.path_ids)
        MSG11 = f"graph_ids are not sorted"
        assert list(self.all_ids) == sorted(list(self.all_ids)), MSG11
        self.unique_ids = sorted(set(self.all_ids))


    def num_nodes(self, i):
        """Gets the number of nodes in a given graph"""

        # get chosen graph id
        chosen_id = self.unique_ids[i]

        # start a counter for number of node
        count_nodes = 0

        # iterate over all_ids increment counter when id = chosen_id
        for ID in self.all_ids:
            if ID == chosen_id:
                count_nodes += 1
        
        return count_nodes

    def counts(self):
        """This function returns the counts of nodes for each graph
        in our dataset"""

        # create a dictionary holding the count of nodes for each graph
        counts = {graph_id: 0 for graph_id in self.unique_ids}
        # iterate over all ids and increment count for that graph id
        for el in self.all_ids:
            counts[el] += 1

        return counts


    def get_shift(self, i):
        """This function calculates a shift so that each graph starts
        with node id 0. Problem lies when i > 0."""

        # Get counts of nodes for each graph in the datset
        counts: dict = self.counts()

        # get chosen id given index i
        chosen_id = self.unique_ids[i]

        # get ids of graphs below chosen id
        # example: if unique_ids = [1,2,3,4,5,6] and chosen_id = 4
        # return [1,2,3]
        ids_below = self.unique_ids[:i]

        # add the number of nodes of each graph below the one we chose
        shift = sum([counts[ID] for ID in ids_below])

        return shift

    # this specifies the indexing of our object
    def get_edges(self, i):
        """ this function will return the edge_index
        array for a single graph indexed by i in our dataset """

        chosen_id = self.unique_ids[i]

        shift = self.get_shift(i)

        # generate an empty list for holding edge_indexes 
        edge_index = []

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

                # get the id of source and target, they must be the same
                source_id = self.all_ids[source]
                target_id = self.all_ids[target]

                # source, target and chosen id must be same
                if source_id == target_id == chosen_id:

                    # get shifted node id
                    source = source - shift
                    target = target - shift

                    # append the shifted source and target link
                    edge_index.append([source, target])

            jsonFile.close()

        edge_index = torch.tensor(edge_index, dtype=torch.long)

        return edge_index.t().contiguous()

    def get_labels(self, i):
        """This function will return the labels for each node
        in graph i"""

        # get shift
        shift: int = self.get_shift(i)
        
        # get chosen_id
        chosen_id: int = self.unique_ids[i]

        # get counts 
        counts: dict = self.counts()

        # load labels (num_nodes, 121)
        labels = torch.from_numpy(np.load(self.path_labels))

        return labels[shift : shift + counts[chosen_id]]

    def get_features(self, i):
        """This function returns the features for graph i"""

        # get filename of features file (if i = 1, fn = 'id_1.pt')
        filename = 'id_' + str(i) + '.pt'

        # construct path to file with features
        path_features = os.path.join(self.path_feats, filename)

        # load the features using torch
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

    g = Graphs('../data', 'train')
    # count_edges = 0
    # for i in range(len(g)):
    #     edges = g.get_edges(i)
    #     count_edges += edges.shape[1]
    # print(count_edges)
    idx = 7
    features = g.get_features(idx)
    labels = g.get_labels(idx)
    print(features.shape)
    print(labels.shape)

