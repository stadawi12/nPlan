import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from loader import dataset

def Train(path_data):
    """This function takes care of training a model

    Parameters
    ----------
    path_data : str
        path to the data directory for example '../data'

    """
    
    # filenames of train features and labels data
    FILENAME_TRAIN_FEATS  = 'train_feats.npy'
    FILENAME_TRAIN_LABELS = 'train_labels.npy'

    # construct path to training data
    path_train_feats  = os.path.join(path_data, FILENAME_TRAIN_FEATS)
    path_train_labels = os.path.join(path_data, FILENAME_TRAIN_LABELS)

    data_train_feats  = dataset(path_train_feats)
    data_train_labels = dataset(path_train_labels)
