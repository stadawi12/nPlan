from loader import dataset
from torch.utils.data import DataLoader
from models import linear
import torch.optim as optim
import torch.nn as nn

def Train(path_data):
    """This function takes care of training a model

    Parameters
    ----------
    path_data : str
        path to the data directory for example '../data'

    """
    
    # Instantiate object of dataset class
    data_train= dataset(path_data, 'train')

    # Initialise data loader with custom batch size and shuffle bool
    loader_train = DataLoader(data_train, batch_size = 1000, shuffle=False)

    # Initialise model
    # TODO allow for option to choose device: 'cpu', 'cuda:0'
    model = linear.Linear()

    # Specify optimiser
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    # bin for storing training losses
    training_losses = []

    # Loss function
    lf = nn.BCELoss()

    for e in range(50):

        training_loss = []

        for feats, labels in loader_train:

            model.zero_grad()
            out = model(feats.float())
            loss = lf(out.float(), labels.float())
            training_loss.append(loss)
            loss.backward()
            optimiser.step()
        print(f"E: {e}, loss: {sum(training_loss)/len(training_loss)}")


if __name__ == '__main__':
    import torch
    torch.manual_seed(0)

    path_data = '../data'

    Train(path_data)
