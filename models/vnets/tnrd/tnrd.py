import tensorflow as tf
from models.vnets.tdv.conv import Conv2d
from optotf.keras.activations import TrainableActivation


class TNRD(tf.keras.Model):
    """Implements TNRD regularizer as detailed in https://arxiv.org/pdf/1508.02848.pdf"""

    def __init__(self,
                 num_channels: int = 1,
                 filters: int = 32,
                 kernel_size: int = 3,
                 **kwargs):
        """Initializes the regularizer

        Parameters
        ----------
        num_channels: (int) - Number of input channels into variational network
        features: (int) - Number of features of convolutions
        kernel_size: (int) - Size of convolution kernels
        kwargs: (int) - tf.keras.Model kwargs
        """
        super(TNRD, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self._build()

    def _build(self):
        self.K = Conv2d(in_channels=self.num_channels,
                        out_channels=self.filters,
                        kernel_size = self.kernel_size,
                        zero_mean = True,
                        bound_norm = True,
                        name = 'K')

        self.act = TrainableActivation(vmin=-0.25,
                                       vmax=0.25,
                                       num_weights=31,
                                       init_scale=0.04)

    def forward(self, x):
        """Computes forward operation of TNRD.
        """
        raise NotImplementedError

    def backward(self, x):
        """Computes backward operation of TNRD i.e. gradient of "forward" operation"""
        x = self.K.forward(x)
        x = self.act(x)
        x = self.K.backward(x)
        x = x / self.filters
        return x

    def call(self, inputs, training=None, mask=None):
        """Using forward and backward methods as calls"""
        raise NotImplementedError

    def grad(self, x):
        """Compute gradient of TDV model"""
        return self.backward(x)

    def get_config(self):
        """Returns configuration of TDV regularizer"""
        config = {
            'num_channels': self.num_channels,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
        }
        base_config = super(TNRD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))