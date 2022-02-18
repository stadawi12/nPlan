from loader import dataset
from torch.utils.data import DataLoader
from models import linear
import torch.optim as optim
import torch.nn as nn

def loadData(path_data):
    pass

def Train(path_data):
    """This function takes care of training a model

    Parameters
    ----------
    path_data : str
        path to the data directory for example '../data'

    """
    
    # Instantiate object of dataset class for training data
    data_train= dataset(path_data, 'train')
    # Initialise data loader with custom batch size and shuffle bool
    loader_train = DataLoader(data_train, batch_size = 1000, shuffle=False)

    # Instantiate object of dataset class
    data_valid= dataset(path_data, 'valid')
    # Initialise data loader with custom batch size and shuffle bool
    loader_valid = DataLoader(data_valid, batch_size = 1000, shuffle=False)

    # Initialise model
    # TODO allow for option to choose device: 'cpu', 'cuda:0'
    model = linear.Linear()
    # Specify optimiser
    optimiser = optim.Adam(model.parameters(), lr=0.001)
    # Loss function
    lf = nn.BCELoss()

    # Begin trining loop
    for e in range(50):

        # Average value of training loss per epoch
        losses_training = []

        # Minibatch loop
        for feats, labels in loader_train:

            # Forward pass
            out = model(feats.float())
            # TODO need to take care of changin things to floats this
            # feels a bit hacky
            loss_training = lf(out.float(), labels.float())

            # append training loss for this batch to training_losses
            losses_training.append(loss_training)

            # Backward pass
            model.zero_grad()
            loss_training.backward()
            optimiser.step()

        # perform diagnostics after each epoch of training
        loss_training_avg = sum(losses_training)/len(losses_training)
        writer.add_scalar("Loss/epoch", loss_training_avg, e)
        print(f"E: {e}, loss: {loss_training_avg}")

        # Average value of validation loss per epoch
        losses_valid = []

        # Validation step for every epoch
        with torch.no_grad():

            for valid_f, valid_l in loader_valid:
                out = model(valid_f.float())
                loss_valid = lf(out.float(), valid_l.float())
                losses_valid.append(loss_valid)
        loss_valid_avg = sum(losses_valid)/len(losses_valid)
        writer.add_scalar("Loss/epoch_v", loss_valid_avg, e)


if __name__ == '__main__':
    import torch
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()

    torch.manual_seed(0)

    path_data = '../data'

    Train(path_data)
    writer.flush()
