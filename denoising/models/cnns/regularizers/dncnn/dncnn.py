from denoising.models.cnns.regularizers.regularizer import Regularizer
from denoising.util import get_num_channels
from typing import Dict
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Subtract
from tensorflow.keras import Input, Model


def DnCNN_Model(depth: int,
                filters: int = 64,
                res: bool = False,
                image_channels: int = 1,
                use_bnorm: bool = True):
    """Constructs DnCNN model similar to https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_keras/main_train.py

    Parameters
    ----------
    depth: (int) - Depth of the DnCNN model
    filters: (int) - Number of filters in each convolutional layer
    res: (bool) - If true, apply residual at end of model. False, otherwise.
    image_channels: (int) - Number of image channels of image
    use_bnorm: (bool) - If True, batch normalization is used in the model
    """
    layer_count = 0
    inpt = Input(shape=(None, None, image_channels), name='input' + str(layer_count))
    # 1st layer, Conv+relu
    layer_count += 1
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
               name='conv' + str(layer_count))(inpt)
    layer_count += 1
    x = Activation('relu', name='relu' + str(layer_count))(x)
    # depth-2 layers, Conv+BN+relu
    for i in range(depth - 2):
        layer_count += 1
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
                   use_bias=False, name='conv' + str(layer_count))(x)
        if use_bnorm:
            layer_count += 1
            # x = BatchNormalization(axis=3, momentum=0.1,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
            x = BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='bn' + str(layer_count))(x)
        layer_count += 1
        x = Activation('relu', name='relu' + str(layer_count))(x)
        # last layer, Conv
    layer_count += 1
    x = Conv2D(filters=image_channels, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal',
               padding='same', use_bias=False, name='conv' + str(layer_count))(x)
    layer_count += 1
    if res:
        x = Subtract(name='subtract' + str(layer_count))([inpt, x])  # input - noise
    model = Model(inputs=inpt, outputs=x)
    return model


class DnCNNRegularizer(Regularizer):
    """Implements DnCNN regularizer as detailed in https://arxiv.org/abs/1608.03981"""

    def __init__(self,
                 config: Dict,
                 **kwargs):
        """Initializes the regularizer

        Parameters
        ----------
        config: (Dict) - Regularizer configuration
        kwargs: (int) - tf.keras.Model kwargs
        """
        super(DnCNNRegularizer, self).__init__(**kwargs)
        self.num_channels = get_num_channels(config)
        self.depth = config['depth']
        self.filters = config['filters']
        self._build()

    def _build(self):
        self.dncnn = DnCNN_Model(depth=self.depth, filters=self.filters)

    def forward(self, x):
        """Computes forward operation of DnCNNRegularizer.
        """
        raise NotImplementedError

    def backward(self, x):
        """Computes backward operation of DnCNNRegularizer i.e. gradient of "forward" operation"""
        x = self.dncnn(x)
        return x

    def grad(self, x, get_energy=False):
        if get_energy:
            raise NotImplementedError
        grad = self.backward(x)
        return grad
