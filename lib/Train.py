from loader import dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import math

# TODO need to write some tests for the Train function

# TODO I wanted to see if I can split the Validation and Testing
# sections of the Train function into their own functions as to avoid
# creating spaghetting code

def Model(name_model: str):
    """Function desgined for choosing different built models for 
    training. Models can be found in 'models' directory. We need to make
    sure that the model is appropriate for our training data, the
    minimum requirement is that a model accepts tensors of shape [m, 50] 
    and outputs tensors of size [m, 121].

    Parameters
    ----------
    name_model : str
        name of model to use, choices so far: ['Linear', 'smallLinear']

    Returns
    -------
    model : torch model
        the wanted model ready for training
    """

    if name_model == 'Linear':
        from models import linear
        return linear.Linear()

    elif name_model == 'smallLinear':
        from models import smallLinear
        return smallLinear.SmallLinear()

    elif name_model == 'normLinear':
        from models import normLinear
        return normLinear.NormLinear()

    elif name_model == 'convNet':
        from models import convNet
        return convNet.UNet()

    elif name_model == 'linRes':
        from models import linRes
        return linRes.LinRes()

def Loss(loss_name: str):
    """ Function designed to allow for choosing different loss functions
    during training. We need to ensure that a loss function is
    appropriate for our training data.
    
    Parameters
    ----------
    loss_name : str
        name of loss function, choices so far: ['BCELoss']

    Returns
    -------
    loss function : torch loss function
        the wanted loss function
    """


    if loss_name == 'BCELoss':
        return nn.BCELoss()


def get_lr(optimiser):
    for param_group in optimiser.param_groups:
        return param_group['lr']

def Train(path_data: str, input_data: dict):
    """This function takes care of training a model

    Parameters
    ----------
    path_data : str
        path to the data directory for example '../data'
    input_data: dict
        a dictionary containig input parameters

    """
    # INPUTS
    # number of examples
    m_train    = input_data["m_train"]
    m_test     = input_data["m_test"]
    m_valid    = input_data["m_valid"]
    # training parameters
    n_epochs   = input_data["n_epochs"]
    batch_size = input_data["batch_size"]
    lr         = input_data["lr"]
    # DataLoader parameters
    shuffle    = input_data["shuffle"]
    num_workers= input_data["num_workers"]
    # device
    device     = input_data["device"]
    device     = torch.device(device)
    # scheduler parameters
    factor     = input_data["factor"]
    patience   = input_data["patience"]
    threshold  = input_data["threshold"]
    # model to use
    model_name = input_data["model"]
    # loss function to use
    loss       = input_data["loss"]

    # intialise tensorboard SummaryWriter for storing training
    # diagnostics
    writer = SummaryWriter("runs/"+model_name)

    # Instantiate object of dataset class for training data
    data_train   = dataset(path_data, 'train', m=m_train)
    # Initialise data loader with custom batch size and shuffle bool
    loader_train = DataLoader(data_train, batch_size=batch_size, 
            shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    # Instantiate object of dataset class for validatiaon data
    data_valid   = dataset(path_data, 'valid', m=m_valid)
    # Initialise data loader with custom batch size and shuffle bool
    loader_valid = DataLoader(data_valid, batch_size=batch_size, 
            shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    # Instantiate object of dataset class for testing data
    data_test   = dataset(path_data, 'test', m_test)
    # Initialise data loader with custom batch size and shuffle bool
    loader_test = DataLoader(data_test, batch_size=batch_size, 
            shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    # TODO allow for option to choose device: 'cpu', 'cuda:0'
    # Initialise model
    model = Model(model_name)
    model.to(device)
    print(model)

    # Specify optimiser
    optimiser = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimiser, factor=factor,
            patience=patience, threshold=threshold)

    # Specify the loss function
    lf = Loss(loss)

    # write model to tensorboard
    writer.add_graph(model, torch.randn(1,50, device=device))

    # Begin trining loop
    for e in range(n_epochs):

        # TRAINING -----------------------------------------------------
        # Average value of training loss per epoch
        losses_training = []

        # initialise batch_number as 0
        batch_number = 0
        # calculate number of batches
        number_of_batches = math.ceil(len(data_train)/batch_size)

        # Minibatch loop
        for feats, labels in loader_train:

            # FORWARD PASS
            # pass training features data through model
            #TODO .to device feels hacky
            out = model(feats.to(device))
            # calculate loss by comparing output with labels
            loss_training = lf(out, labels.to(device))

            # append training loss for this batch to training_losses
            losses_training.append(loss_training)

            # BACKWARD PASS
            model.zero_grad()
            loss_training.backward()
            optimiser.step()
            batch_number += 1
            # if model_name == 'convNet':
            #     print(f"Batch: {batch_number}/{number_of_batches}")


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
                out = model(valid_f.to(device))
                # calculate loss
                loss_valid = lf(out, valid_l.to(device))
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
                out = model(test_f.to(device))

                # binarise the output, (y_i>0.5)->1, (y_j<=0.5)->0
                out = torch.where(out>0.5, 1., 0.)

                # check each sample of model output against test labels
                for i in range(out.shape[0]):

                    # check how many predictions are correct and
                    # increment counter for every correct prediction
                    if torch.equal(out[i], test_l[i].to(device)):

                        # increment counter
                        counter_correct += 1

            # After passing all test data through the model add accuracy
            # to tensorboard
            # percentage_correct = counter_correct / len(data_test)
            writer.add_scalar("accuracy", counter_correct, e)

        # END OF EPOCH -------------------------------------------------
        scheduler.step(loss_training_avg)
        random_inpt, hello  = next(iter(loader_test))
        random_out = model(random_inpt.to(device))
        print(random_out)

        # At the end of each epoch print a statement to the console
        print(f"E: {e}, loss: {loss_training_avg:.5f}, Accuracy: " +
                f"{counter_correct}/{len(data_test)}, " +
                f"lr={get_lr(optimiser)}")

    # finish tensorboard writing
    writer.flush()

if __name__ == '__main__':
    import torch
    from Inputs import Read_Input

    path_input = '../inputs.yaml'
    path_data = '../data'

    input_data = Read_Input(path_input)

    torch.manual_seed(0)

    Train(path_data, input_data)
