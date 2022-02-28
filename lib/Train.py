from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import math
from loadModel import LoadModel, LoadLoss

def get_lr(optimiser):
    # function for obtaining learning rate during training, the 
    # learning rate changes during training and we want to be able
    # to see how it changes
    for param_group in optimiser.param_groups:
        return param_group['lr']

def Train(train_x, train_y, test_x, test_y, input_data):
    """This function takes care of training a model

    Parameters
    ----------
    path_data : str
        path to the data directory for example '../data'
    input_data: dict
        a dictionary containig input parameters

    """
    # INPUTS
    n_epochs   = input_data["n_epochs"]
    lr         = input_data["lr"]
    device     = input_data["device"]
    device     = torch.device(device)
    factor     = input_data["factor"]
    patience   = input_data["patience"]
    threshold  = input_data["threshold"]
    model_name = input_data["model"]
    loss       = input_data["loss"]
    record_run = input_data["record_run"]
    save_model = input_data["save_model"]
    dir_name   = input_data["dir_name"]


    # intialise tensorboard SummaryWriter for storing training
    # diagnostics
    if record_run:
        writer = SummaryWriter(comment=model_name)

    # Initialise model
    model = LoadModel(model_name)
    model.to(device)

    # Specify optimiser
    optimiser = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimiser, factor=factor,
            patience=patience, threshold=threshold)

    # Specify the loss function
    lf = LoadLoss(loss)

    # write model to tensorboard
    if record_run:
        writer.add_graph(model, torch.randn(1,50, device=device))

    # Begin trining loop
    for e in range(n_epochs):

        # TRAINING -----------------------------------------------------

        # FORWARD PASS
        # pass training features data through model
        out = model(train_x.to(device))
        # calculate loss by comparing output with labels
        loss_training = lf(out, train_y.to(device))

        # BACKWARD PASS
        model.zero_grad()
        loss_training.backward()
        optimiser.step()

        # add the average loss to tensor board
        if record_run:
            writer.add_scalar("t_loss/epoch", loss_training, e)

        # VALIDATION ---------------------------------------------------

        # Validation step for every epoch
        with torch.no_grad():
            
            # pass batch of validation data through networ
            out = model(test_x.to(device))
            # calculate loss
            loss_valid = lf(out, test_y.to(device))

        # add data point to tensorboard
        if record_run:
            writer.add_scalar("v_loss/epoch", loss_valid, e)

        # ACCURACY -----------------------------------------------------
        # Test accuracy of network for each epoch, calculate the number
        # of correct predictions
        with torch.no_grad():
            
            # initialise counter of correct predictions at start of 
            # each epoch to 0
            counter_correct = 0

            # pass test batch through the model
            out = model(test_x.to(device))

            # binarise the output, (y_i>0.5)->1, (y_j<=0.5)->0
            out = torch.where(out>0.5, 1., 0.)

            # check each sample of model output against test labels
            for i in range(out.shape[0]):

                # check how many predictions are correct and
                # increment counter for every correct prediction
                if torch.equal(out[i], test_y[i].to(device)):

                    # increment counter
                    counter_correct += 1

            acc = counter_correct / len(test_x) * 100
            # After passing all test data through the model add accuracy
            # to tensorboard
            # percentage_correct = counter_correct / len(data_test)
            if record_run:
                writer.add_scalar("accuracy", acc, e)

        # END OF EPOCH -------------------------------------------------
        scheduler.step(loss_training)

    # At the end of training epoch print a statement to the console
    print(f"E: {e}, loss: {loss_training:.5f}, Accuracy: " +
            f"{acc:.5f}, " +
            f"lr={get_lr(optimiser)}")

    # finish tensorboard writing
    metric_dict = {"loss": loss_training, 
                   "accuracy": acc}
  
    # Add params hyperparameters and close tensorboard writing
    if record_run:
        writer.add_hparams(input_data, metric_dict)
        writer.flush()

    # Save model if save_model is set to true
    if save_model:
        # import datetime and os modules
        import datetime
        import os
        # get todays date
        now = datetime.datetime.now()
        # transform into a readable format
        DATE  = now.strftime("%b%y_%H-%M-%S")

        # construct the name of the model (tail)
        TAIL = f"{DATE}_{model_name}_e{n_epochs}.pt"

        # join directory name with name of model (tail)
        PATH_FULL = os.path.join(dir_name, TAIL)

        # save model
        torch.save(model.state_dict(), PATH_FULL)

if __name__ == '__main__':
    import torch
    from Inputs import Read_Input
    from graph import Graphs

    path_input = '../inputs.yaml'
    path_data = '../data'

    input_data = Read_Input(path_input)

    data = Graphs(path_data, 'train')

    for i in range(len(data)):

        features = data.get_features(i)
        labels = data.get_labels(i)

        train_x = features[:-170].float()
        train_y = labels[:-170].float()
        test_x = features[-170:].float()
        test_y = labels[-170:].float()

        Train(train_x, train_y, test_x, test_y, input_data)
