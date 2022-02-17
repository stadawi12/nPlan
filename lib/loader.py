from torch.utils.data import Dataset
import numpy as np

# TODO need to write tests for this module and find out if loading the
# entire dataset when instantiating a dataset object leads to memory
# issues

class dataset(Dataset):
    """ Class that takes care of loading data,
    we want to be able to load train, test or validation data.

    Need to define __len__() function and the __getitem__() function."""

    def __init__(self, path_data):

        """
        Parameters
        ---------
        path_data : str
            full path to the dataset we want to load, for example
            '../data/train_feats.npy' 
        """
        
        self.path_data = path_data

        # load the dataset
        self.data = np.load(path_data)


    # controls the behaviour of the len() method
    def __len__(self):
        return len(self.data)

    # controls the behaviour of indexing dataset()[i]
    def __getitem__(self, i):
        return self.data[i]


if __name__ == '__main__':
    # Test to see if the dataset class works as intended
    from torch.utils.data import DataLoader

    # Path to train_feats.npy file with node features data
    path = '../data/train_feats.npy'
    # Instantiate object of dataset class
    tf = dataset(path)

    # Initialise data loader with custom batch size and shuffle bool
    data_loader = DataLoader(tf, batch_size = 21, shuffle=False)

    # iterate over batches of data and check if shape of data agrees for
    # the first 10 batches
    for i, node in enumerate(data_loader):
        # Ensure shape is correct
        print(node.shape)
        # Ensure the data type is a Tensor
        print(type(node))
        # After 10th batch, break
        if i == 10:
            break
