import os
from torch.utils.data import Dataset
import numpy as np

# TODO need to write tests for this module and find out if loading the
# entire dataset when instantiating a dataset object leads to memory
# issues

class dataset(Dataset):
    """ Class that takes care of loading data,
    we want to be able to load train, test or validation data.

    Need to define __len__() function and the __getitem__() function."""

    def __init__(self, path_data: str, data_for: str):

        """
        Parameters
        ---------
        path_data : str
            path to data directory for example '../data' 
        data_for : str
            choose to load 'train', 'test' or 'valid' datasets of
            features and labels
        """
        
        # path to global data directory
        self.path_data: str = path_data
        # name of dataset, 'train', 'test' or 'valid'
        self.data_for: str  = data_for

        # generate full paths to data
        path_feats: str  = os.path.join(self.path_data, 
                self.data_for+'_feats.npy')
        path_labels: str = os.path.join(self.path_data, 
                self.data_for+'_labels.npy')

        # load the features and lables for a given dataset
        self.data_feats = np.load(path_feats)
        self.data_labels = np.load(path_labels)

        # TODO assert that length of data_feats is same as length
        # data_labels


    # controls the behaviour of the len() method
    def __len__(self):
        return len(self.data_feats)

    # controls the behaviour of indexing dataset()[i]
    def __getitem__(self, i):
        return self.data_feats[i], self.data_labels[i]


if __name__ == '__main__':
    # Test to see if the dataset class works as intended
    from torch.utils.data import DataLoader

    # Path to train_feats.npy file with node features data
    path = '../data'
    # Instantiate object of dataset class
    tf = dataset(path, 'train')

    # Initialise data loader with custom batch size and shuffle bool
    data_loader = DataLoader(tf, batch_size = 21, shuffle=False)

    # iterate over batches of data and check if shape of data agrees for
    # the first 10 batches
    for i, node in enumerate(data_loader):
        feats, labels = node
        # Ensure shape is correct
        print(feats.shape)
        print(labels.shape)
        # Ensure the data type is a Tensor
        print(type(feats))
        print(type(labels))
        # After 10th batch, break
        if i == 10:
            break
