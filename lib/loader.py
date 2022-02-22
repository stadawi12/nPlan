import os
from torch.utils.data import Dataset
import numpy as np
import torch

# TODO need to write tests for this module and find out if loading the
# entire dataset when instantiating a dataset object leads to memory
# issues

class dataset(Dataset):
    """ Class that takes care of loading data,
    we want to be able to load train, test or validation data.

    Need to define __len__() function and the __getitem__() function."""

    def __init__(self, path_data: str, data_for: str, m: int = None,
            device='cpu'):

        """
        Parameters
        ---------
        path_data : str
            path to data directory for example '../data' 
        data_for : str
            choose to load 'train', 'test' or 'valid' datasets of
            features and labels
        m : int
            number of examples to load
        """

        self.device = device
        self.device = torch.device(self.device)
        
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
        # if m is specified, only load m examples, if m is None, 
        # load all examples
        if m == None:
            self.data_feats = np.load(path_feats)
            self.data_labels = np.load(path_labels)
        else:
            self.data_feats = np.load(path_feats)
            self.data_labels = np.load(path_labels)

            self.data_feats = np.load(path_feats)[:m]
            self.data_labels = np.load(path_labels)[:m]

        # Assert that length of features and labels has to be the same
        assert self.data_feats.shape[0] == self.data_labels.shape[0], \
                "Length of data is not equal size"

        # convert data to torch tensor


    # controls the behaviour of the len() method
    def __len__(self):
        return len(self.data_feats)

    # controls the behaviour of indexing dataset(..)[i]
    def __getitem__(self, i):
        tensor_feature = torch.from_numpy(self.data_feats[i]).float()
        tensor_label = torch.from_numpy(self.data_labels[i]).float()
        return tensor_feature.to(self.device), tensor_label.to(self.device)


if __name__ == '__main__':
    # Test to see if the dataset class works as intended
    from torch.utils.data import DataLoader

    # Path to train_feats.npy file with node features data
    path = '../data'
    # Instantiate object of dataset class
    tf = dataset(path, 'train', m=1000, device='cuda:0')
    print(len(tf))

    # Initialise data loader with custom batch size and shuffle bool
    data_loader = DataLoader(tf, batch_size = 1, shuffle=False)

    # iterate over batches of data and check if shape of data agrees for
    # the first 10 batches
    for i, node in enumerate(data_loader):
        feats, labels = node
        # Ensure shape is correct
        print(feats.shape)
        print(labels.shape)
        print(feats.device)
        print(labels.dtype)
        # print(feats)
        # print(labels)
        # Ensure the data type is a Tensor
        print(type(feats))
        print(type(labels))
        # After 10th batch, break
        if i == 0:
            break
