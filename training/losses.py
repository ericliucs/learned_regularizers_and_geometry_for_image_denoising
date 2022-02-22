import tensorflow.keras.backend as K
from typing import Dict
from tensorflow.keras.losses import MeanSquaredError

def retrieve_loss_function(config: Dict):
    """Retrieves loss function by name

    Parameters
    ----------
    config: (Dict) - Configuration that contains loss_function key

    Returns
    -------

    """
    if config['loss_function'] == 'sum_squared_error_loss':
        return sum_squared_error_loss
    else:
        raise Exception('The loss of the model was not specified correctly')


def sum_squared_error_loss(clean,x):
    """ Defines sum squared error loss

    Parameters
    ----------
    clean
    x

    Returns
    -------

    """
    loss = K.sum(K.square(x - clean))/2
    return loss