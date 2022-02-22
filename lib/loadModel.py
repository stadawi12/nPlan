# - loadModel.py

def LoadModel(model_name: str, batch_norm: bool = False):
    """Function desgined for choosing different built models for 
    training. Models can be found in 'models' directory. We need to make
    sure that the model is appropriate for our training data, the
    minimum requirement is that a model accepts tensors of shape [m, 50] 
    and outputs tensors of size [m, 121].

    Parameters
    ----------
    model_name : str
        name of model to use, choices so far: ['Linear', 'smallLinear']
    batch_norm : bool
        a boolean value which determines whether you want to use 
        batch_norm or not

    Returns
    -------
    model : torch model
        the wanted model ready for training
    """

    # List of available models
    available_models = [
            'Linear', 
            'smallLinear', 
            'linRes', 
            'linResBN' 
            ]

    # assert that the model we have chosen is inside the list of 
    # available models
    assert model_name in available_models, f"The model {model_name} does not exist"

    if model_name == 'Linear':
        from models import linear
        return linear.Linear()

    elif model_name == 'smallLinear':
        from models import smallLinear
        return smallLinear.SmallLinear()

    elif model_name == 'linRes':
        from models import linRes
        return linRes.LinRes()

    elif model_name == 'linResBN':
        from models import linRes
        return linRes.LinRes()

def LoadLoss(loss_name: str):
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
    # List of available losses
    available_losses =['BCELoss']

    # assert that the loss we have chosen is inside the list of 
    # available losses
    assert loss_name in available_losses, f"The loss {loss_name} does not exist"

    if loss_name == 'BCELoss':
        import torch.nn as nn
        return nn.BCELoss()
