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

        # create filename of the feats.npy file
        self.filename_feats = self.dataset + '_feats.npy'

        # create path to feats.npy file and assert it exists
        self.path_feats = os.path.join(self.path_data,
                self.filename_feats)
        MSG5 = f"Path {self.path_feats} does not exist"
        assert os.path.exists(self.path_feats), MSG5

        # create filename of the labels.npy file
        self.filename_labels = self.dataset + '_labels.npy'
        # create path to labels.npy file and assert it exists
        self.path_labels = os.path.join(self.path_data,
                self.filename_labels)
        MSG6 = f"Path {self.path_labels} does not exist"
        assert os.path.exists(self.path_labels), MSG6

    # this specifies the indexing of our object
    def __getitem__(self, i):
        """ using the index we will return an features and edge_index
        array for a single graph indexed by i in our dataset """

        # load graph_id.npy file as a numpy array
        # this file contains the graph ids that each node belongs to
        # series of [1,1,1,...,2,2,2,...,3,3,.....]
        graph_ids = np.load(self.path_ids)

        MSG11 = f"graph_ids are not sorted"
        assert list(graph_ids) == sorted(list(graph_ids)), MSG11

        # load features (num_nodes, 50)
        features = torch.from_numpy(np.load(self.path_feats))

        # load labels (num_nodes, 121)
        labels = torch.from_numpy(np.load(self.path_labels))

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

        edge_index = torch.tensor(edge_index, dtype=torch.long)

        return (features[shift : shift + counts[chosen_graph_id]], 
                edge_index.t().contiguous(), 
                labels[shift : shift + counts[chosen_graph_id]])

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

    class Node2vec(Node2Vec):

        def test(self):
            print("Hello! I am captain now")


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


    g = Graphs('../data', 'train')
    x, edge_index, labels = g[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(edge_index, embedding_dim=50, walk_length=20,
            context_size=10, walks_per_node=20, 
            num_negative_samples=2, p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=False, num_workers=0)
    optimiser = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


    for epoch in range(1,101):
        loss = train()
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

    z = model()
    torch.save(z.detach().cpu(), 'features.pt')
    torch.save(labels, 'labels.pt')
