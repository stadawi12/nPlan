
def LoadModel(name_model: str, batch_norm: bool):
    """Function desgined for choosing different built models for 
    training. Models can be found in 'models' directory. We need to make
    sure that the model is appropriate for our training data, the
    minimum requirement is that a model accepts tensors of shape [m, 50] 
    and outputs tensors of size [m, 121].

    Parameters
    ----------
    name_model : str
        name of model to use, choices so far: ['Linear', 'smallLinear']
    batch_norm : bool
        a boolean value which determines whether you want to use 
        batch_norm or not

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

    elif name_model == 'linRes':
        from models import linRes
        return linRes.LinRes(norm=batch_norm)

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


    if loss_name == 'BCELoss':
        return nn.BCELoss()
