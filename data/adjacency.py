import numpy as np
import json
import matplotlib.pyplot as plt 
import os

class Adjacency():

    """Class to extract the adjacency matrix for a specific graph"""

    def __init__(self, path_data: str, dataset: str):
        """Parameters
        ----------
        path_data : str
            this is the path to the data directory where all graph data is
            stored, for example, if you are running this function from the
            main directory you would specify path_data = 'data', the default
            is '' (empty) meaning we are already in the data directory
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

    # this specifies the indexing of our object
    def __getitem__(self, i):
        """ using the index we will return an adjacency matrix for a
        single graph in our dataset """

        # load graph_id.npy file as a numpy array
        # this file contains the graph ids that each node belongs to
        # series of [1,1,1,...,2,2,2,...,3,3,.....]
        graph_ids = np.load(self.path_ids)

        # calculate how many nodes in each graph using dictionary
        # counting, counts = {graph_id: count}
        counts = {x: 0 for x in set(graph_ids)}
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

        # if chosen_graph_id is larger than smallest_graph_id then
        # specify a shift. Shift is needed because we only are interested 
        # in a single adjacency matrix of graph_i which is a particular
        # block in the total adjacency matrix of all graphs
        #    ((A1)        )
        #A = (    (A2)    )
        #    (        (A3))
        # If we just want A3 which is the adjacency matrix of graph_id 3
        # then we need to shift the row and column indices by the size
        # of A1 and A2, the following will determine the necessary
        # shift.
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

        # get the matrix size of our adjacency matrix
        matrix_size = counts[chosen_graph_id]

        # generate an empty adjacency matrix which we will populate
        A = np.zeros((matrix_size, matrix_size))

        # opent the json file with all edges between nodes for our
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
                    A[source - shift, target - shift] += 1

            jsonFile.close()

        return A

    def __len__(self):
        graph_ids = np.load(self.path_ids)

        counts = {x: 0 for x in set(graph_ids)}
        for el in graph_ids:
            counts[el] += 1

        return len(counts)

if __name__ == '__main__':
    a = Adjacency('.', 'train')

    # We want to check if the number total links is equal to the 
    # sum of all adjacency matrices that we construct
    # this will ensure that we at least do not miss a single link if the
    # two numbers are the same.
    with open(a.path_json) as jsonFile:
        jsonObject = json.load(jsonFile)

        # get all links
        links = jsonObject['links']
        # obtain the length of links, i.e. the number of all links for
        # all graphs in a dataset
        n_link_actual = len(links) 
        
        # run a for loop constructing adjacency matrices for all graphs
        # in dataset and then counting the sum of each adjacency matrix
        # and adding that to a count, n_link_actual should be equal to
        # count_links
        count_links = 0
        for i in range(len(a)):
            print(f"Constructing adjacency matrix for graph: {i}")
            # generate an adjacency matrix for a single graph
            x = a[i]
            # calculate the sum of all entries of the adjacency matrix
            # which is equal to the number of links in that graph and
            # add that to the count (count_links)
            count_links += np.sum(x)

        # once iterated through all graphs in a dataset check that the
        # two numbers are the same
        print(n_link_actual)
        print(count_links)
        if n_link_actual == count_links:
            print("We haven't missed any links")

        jsonFile.close()
