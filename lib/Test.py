from loadModel import LoadModel
import torch
from dataloader import dataset
from Inputs import Read_Input
import os


def Test(path_data: str, path_models:str, input_data: dict):
    """This module is designed to test a pretrained model
    on a custom data set, the dataset must be provided as
    a .npy file of m number of examples and each example
    must be 50 features long.

    Parameters
    ----------
    path_data : str
        path to directory where .npy file containing test data
        is stored
    path_models : str
        path to directory where the model is stored
    input_data : dict
        this is a dictionary containing some model parameters
        required for using the model

    """
    # INPUTS
    NAME_MODEL:str    = input_data["name_model"]
    NAME_TRAINED:str  = input_data["name_trained"]
    NAME_FEATURES:str = input_data["name_features"]
    NAME_LABELS:str   = input_data["name_labels"]
    BATCH_NORM:bool   = input_data["batch_norm"]

    # construct path to feature and label data
    path_features = os.path.join(path_data, NAME_FEATURES)
    path_labels   = os.path.join(path_data, NAME_LABELS)

    # load features and labels data as numpy arrays
    data_features = np.load(path_features)
    data_labels   = np.load(path_labels)

    # convert data from numpy to tensor and convert dtype to float
    data_features = torch.from_numpy(data_features).float()
    data_labels   = torch.from_numpy(data_labels).float()

    # construct path to pretrained model
    path_trained:str = os.path.join(path_models, NAME_TRAINED)

    # load model
    model = LoadModel(NAME_MODEL)
    model.load_state_dict(torch.load(path_trained))
    model.eval()

    # pass feature data through model
    out = model(data_features)
    out = torch.where(out>0.5, 1., 0.)

    # initialise counter of correct predictions at start of 
    # each epoch to 0
    counter_correct = 0
    # check each sample of model output against test labels
    for i in range(out.shape[0]):

        # check how many predictions are correct and
        # increment counter for every correct prediction
        if torch.equal(out[i], data_labels[i]):

            # increment counter
            counter_correct += 1

    print(counter_correct)



if __name__ == "__main__":
    pass
