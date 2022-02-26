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

        # create filename of the graph.json file
        self.filename_feats = self.dataset + '_feats.npy'

        # create path to graph.json file and assert it exists
        self.path_feats = os.path.join(self.path_data,
                self.filename_feats)
        MSG4 = f"Path {self.path_feats} does not exist"
        assert os.path.exists(self.path_feats), MSG4

    # this specifies the indexing of our object
    def __getitem__(self, i):
        """ using the index we will return an features and edge_index
        array for a single graph indexed by i in our dataset """

        # load graph_id.npy file as a numpy array
        # this file contains the graph ids that each node belongs to
        # series of [1,1,1,...,2,2,2,...,3,3,.....]
        graph_ids = np.load(self.path_ids)

        # load features (num_nodes x 50)
        features = np.load(self.path_feats)

        MSG5 = f"graph_ids are not sorted"
        assert list(graph_ids) == sorted(list(graph_ids)), MSG5

        # calculate how many nodes in each graph using dictionary
        # counting, counts = {graph_id: count}
        counts = {x: 0 for x in sorted(set(graph_ids))}
        for el in graph_ids:
            counts[el] += 1
        
        # obtain the keys of counts dictionary, e.g. keys = [1,2,3,4,5,...]
        keys = counts.keys()

        # extract chosen_graph_id, given index = 0, which graph id are we
        # asking for, that's what chosen_graph_id is
        chosen_graph_id = list(keys)[i]

        # get the smallest graph id in our dataset, if list of graph ids
        # is like [1,2,3,4,...], smallest_graph_id would be 1
        smallest_graph_id = min(list(keys))

        # this specifies the shift to adjacency matrix, if
        # chosen_graph_id is the same as smallest_graph_id then there is
        # no shift.         
        shift = 0

        if chosen_graph_id > smallest_graph_id:
            # get ids of all graphs which are below the one we are
            # interested in
            ids_above = []
            for key in sorted(list(keys)):
                if key < chosen_graph_id:
                    ids_above.append(key)
            # shift is equal to the sum of sizes of adjacency matrices
            # that are above it
            shift = sum([counts[ID] for ID in ids_above])

        x = features[shift : shift + counts[chosen_graph_id]]

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

                # check that the graph_id of source and target are the
                # same as the graph_id that we are interested in 
                if graph_ids[source] == graph_ids[target] == chosen_graph_id:

                    # update the adjacency to include the link
                    # information
                    edge_index.append([source, target])

            jsonFile.close()

        x = torch.from_numpy(x)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        return x, edge_index.t().contiguous()

    def __len__(self):
        # return the number of graphs found in dataset
        graph_ids = np.load(self.path_ids)

        ids = set(graph_ids)

        return len(ids)

if __name__ == '__main__':
    from torch_geometric.nn import Node2Vec
    import torch
    try:
        import torch_cluster  # noqa
        random_walk = torch.ops.torch_cluster.random_walk
    except ImportError:
        random_walk = None

    g = Graphs('.', 'train')
    x, edge_index = g[0]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(edge_index, embedding_dim=50, walk_length=20,
            context_size=10, walks_per_node=20, 
            num_negative_samples=1, p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=False, num_workers=0)
    print(len(loader))
    optimiser = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimiser.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        return total_loss / len(loader)


    for epoch in range(1,11):
        loss = train()
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
    # for idx, (pos_rw, neg_rw) in enumerate(loader):
        # print(idx, pos_rw.shape, neg_rw.shape)
    # print(pos_rw[:,0])
    # print(neg_rw[0])
    # print(pos_rw.shape)
    # print(neg_rw.shape)
    # print(b.shape 

    # features = np.load(g.path_feats)
    # len_features = len(features)

    # # We want to check if the number total links is equal to the 
    # # sum of all adjacency matrices that we construct
    # # this will ensure that we at least do not miss a single link if the
    # # two numbers are the same.
    # with open(g.path_json) as jsonFile:
        # jsonObject = json.load(jsonFile)

        # # get all links
        # links = jsonObject['links']
        # # obtain the length of links, i.e. the number of all links for
        # # all graphs in a dataset
        # n_link_actual = len(links) 

        # links_count = 0
        # features_count = 0

        # for i in range(len(g)):
            # print(i)
            # x, edge_index = g[i]
            # links_count += len(edge_index)
            # features_count += len(x)

        # print(n_link_actual)
        # print(links_count)
        # if n_link_actual == links_count:
            # print("We haven't missed any links")

        # print(len_features)
        # print(features_count)
        # if len_features == features_count:
            # print("We haven't missed any features")

        # jsonFile.close()
