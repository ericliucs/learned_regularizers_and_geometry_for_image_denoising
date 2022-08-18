from typing import Dict
import tensorflow as tf
from denoising.models.cnns.regularizers.tv.tv import TVRegularizer


def retrieve_loss_function(config: Dict):
    """Retrieves loss function by name

    Parameters
    ----------
    config: (Dict) - Configuration that contains loss_function key

    """
    if config['loss_function'] == 'sum_squared_error_loss':
        return sum_squared_error_loss
    elif config['loss_function'] == 'mean_sum_squared_error_loss':
        return mean_sum_squared_error_loss
    elif config['loss_function'] == 'mean_sum_squared_error_curvature_loss':
        curvature_loss = CurvatureLoss()
        return curvature_loss.mean_sum_squared_error_curvature_loss
    elif config['loss_function'] == 'mean_sum_absolute_error_curvature_loss':
        curvature_loss = CurvatureLoss()
        return curvature_loss.mean_sum_absolute_error_curvature_loss
    else:
        raise Exception('The loss of the model was not specified correctly')


def mean_sum_squared_error_loss(clean, x):
    """ Defines mean sum squared error loss"""
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - clean), axis=[1, 2, 3]) / 2)
    return loss


class CurvatureLoss:
    """Defines curvature loss"""

    def __init__(self):
        """Initializes Keras model to compute curvature"""
        self.curvature = TVRegularizer({}).grad

    def mean_sum_squared_error_curvature_loss(self, clean, x):
        """Defines mean sum squared error curvature loss"""
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.curvature(x) - self.curvature(clean)), axis=[1, 2, 3]) / 2)
        return loss

    def mean_sum_absolute_error_curvature_loss(self, clean, x):
        """Defines mean sum absolute error curvature loss"""
        loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.curvature(x) - self.curvature(clean)), axis=[1, 2, 3]))
        return loss


def sum_squared_error_loss(clean, x):
    """Defines sum squared error loss"""
    loss = tf.reduce_sum(tf.square(x - clean))/2
    return loss
