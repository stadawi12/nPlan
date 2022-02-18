from loader import dataset
from torch.utils.data import DataLoader
from models import linear
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_lr(optimiser):
    for param_group in optimiser.param_groups:
        return param_group['lr']

def Train(path_data):
    """This function takes care of training a model

    Parameters
    ----------
    path_data : str
        path to the data directory for example '../data'

    """
    # intialise tensorboard SummaryWriter for storing training
    # diagnostics
    writer = SummaryWriter()

    # Instantiate object of dataset class for training data
    data_train   = dataset(path_data, 'train')
    # Initialise data loader with custom batch size and shuffle bool
    loader_train = DataLoader(data_train, batch_size = 1000, shuffle=False)

    # Instantiate object of dataset class for validatiaon data
    data_valid   = dataset(path_data, 'valid')
    # Initialise data loader with custom batch size and shuffle bool
    loader_valid = DataLoader(data_valid, batch_size = 1000, shuffle=False)

    # Instantiate object of dataset class for testing data
    data_test   = dataset(path_data, 'test')
    # Initialise data loader with custom batch size and shuffle bool
    loader_test = DataLoader(data_test, batch_size = 1000, shuffle=False)

    # Initialise model
    # TODO allow for option to choose device: 'cpu', 'cuda:0'
    model = linear.Linear()
    # Specify optimiser
    optimiser = optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimiser, factor=0.5, patience=4,
            threshold=0.001)
    # Loss function
    lf = nn.BCELoss()
    # write model to tensorboard
    writer.add_graph(model, torch.randn(1,50))

    # Begin trining loop
    for e in range(50):

        # TRAINING -----------------------------------------------------
        # Average value of training loss per epoch
        losses_training = []

        # Minibatch loop
        for feats, labels in loader_train:

            # FORWARD PASS
            # pass training features data through model
            out = model(feats.float())
            # TODO need to take care of changin things to floats this
            # feels a bit hacky
            # calculate loss by comparing output with labels
            loss_training = lf(out.float(), labels.float())

            # append training loss for this batch to training_losses
            losses_training.append(loss_training)

            # BACKWARD PASS
            model.zero_grad()
            loss_training.backward()
            optimiser.step()

        # perform diagnostics after each training epoch
        # calculate loss average of epoch
        loss_training_avg = sum(losses_training)/len(losses_training)
        # add the average loss to tensor board
        writer.add_scalar("t_loss/epoch", loss_training_avg, e)

        # VALIDATION ---------------------------------------------------
        # Average value of validation loss per epoch
        losses_valid = []

        # Validation step for every epoch
        with torch.no_grad():
            
            # pass validation dataset through network and calculate the
            # loss
            for valid_f, valid_l in loader_valid:

                # pass batch of validation data through networ
                out = model(valid_f.float())
                # calculate loss
                loss_valid = lf(out.float(), valid_l.float())
                # append loss to losses bin
                losses_valid.append(loss_valid)

        # calculate average of validation loss for all validation data
        # in this epoch
        loss_valid_avg = sum(losses_valid)/len(losses_valid)
        # add data point to tensorboard
        writer.add_scalar("v_loss/epoch", loss_valid_avg, e)

        # ACCURACY -----------------------------------------------------
        # Test accuracy of network for each epoch, calculate the number
        # of correct predictions
        with torch.no_grad():
            
            # initialise counter of correct predictions at start of 
            # each epoch to 0
            counter_correct = 0

            # pass a batch of test data through the model
            # test_f = test data features
            # test_l = test data labels
            for test_f, test_l in loader_test:

                # pass test batch through the model
                out = model(test_f.float())

                # binarise the output, (y_i>0.5)->1, (y_j<=0.5)->0
                out = torch.where(out>0.5, 1, 0)

                # check each sample of model output against test labels
                for i in range(out.shape[0]):

                    # check how many predictions are correct and
                    # increment counter for every correct prediction
                    if torch.equal(out[i], test_l[i]):

                        # increment counter
                        counter_correct += 1

            # After passing all test data through the model add accuracy
            # to tensorboard
            # percentage_correct = counter_correct / len(data_test)
            writer.add_scalar("accuracy", counter_correct, e)

        # END OF EPOCH -------------------------------------------------
        scheduler.step(loss_training_avg)

        # At the end of each epoch print a statement to the console
        print(f"E: {e}, loss: {loss_training_avg}, Accuracy: " +
                f"{counter_correct}/{len(data_test)}, " +
                f"lr={get_lr(optimiser)}")

    # finish tensorboard writing
    writer.flush()

if __name__ == '__main__':
    import torch

    torch.manual_seed(0)

    path_data = '../data'

    Train(path_data)
