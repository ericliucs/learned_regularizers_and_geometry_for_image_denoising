from denoising.models.cnns.regularizers.tdv.conv import Conv2d
from optotf.keras.activations import TrainableActivation
from denoising.models.cnns.regularizers.regularizer import Regularizer
import tensorflow_probability as tfp
import tensorflow as tf
from denoising.util import get_num_channels
from typing import Dict


class TNRDRegularizer(Regularizer):
    """Implements TNRD regularizer as detailed in https://arxiv.org/pdf/1508.02848.pdf"""

    def __init__(self,
                 config: Dict,
                 **kwargs):
        """Initializes the regularizer

        Parameters
        ----------
        config: (Dict) - Regularizer configuration
        kwargs: (int) - tf.keras.Model kwargs
        """
        super(TNRDRegularizer, self).__init__(**kwargs)
        self.num_channels = get_num_channels(config)
        self.filters = config['filters']
        self.kernel_size = config['kernel_size']
        self._build()

    def _build(self):
        """
        Builds convolution and RBF activation layers.
        """
        self.K = Conv2d(in_channels=self.num_channels,
                        out_channels=self.filters,
                        kernel_size=self.kernel_size,
                        zero_mean=True,
                        bound_norm=True,
                        name='K')

        self.act = TrainableActivation(vmin=-0.25,
                                       vmax=0.25,
                                       num_weights=31,
                                       init_scale=0.04)

    def act_integral(self, x):
        """Numerically computes integral of RBF activation functions
        """
        y = tf.expand_dims(tf.linspace(-0.5, 0.5, 10000), axis=1)
        y = tf.tile(y, [1, self.filters])
        y_act = self.act(y)
        rho = tf.cumsum(y_act)
        y_ref = rho*(y[-1] - y[0]) / (int(y.shape[0]) - 1)
        for i in range(self.filters):
            if i == 0:
                output = tf.expand_dims(tfp.math.interp_regular_1d_grid(
                x[:,:,:,i], x_ref_min=-0.5, x_ref_max=0.5, y_ref=y_ref[:,i], axis=-1,fill_value_below=0,
                    fill_value_above=0), axis=3)
            else:
                m = tf.expand_dims(tfp.math.interp_regular_1d_grid(
                        x[:,:,:,i], x_ref_min=-0.5, x_ref_max=0.5, y_ref=y_ref[:,i], axis=-1,fill_value_below=0,
                        fill_value_above=0), axis=3)
                output = tf.concat([output,m], axis=3)
        return output

    def forward(self, x):
        """Approximates forward operation of TNRDRegularizer based on backward operation.
        """
        x = self.K.forward(x)
        x = self.act_integral(x)
        x = tf.reduce_sum(x, axis=-1, keepdims=True)
        return x

    def backward(self, x):
        """Computes backward operation of TNRDRegularizer."""
        x = self.K.forward(x)
        x = self.act(x)
        x = self.K.backward(x)
        return x

    def grad(self, x, get_energy=False):
        if get_energy:
            raise NotImplementedError
        grad = self.backward(x)
        return grad / self.filters


if __name__ == "__main__":
    """For Testing"""
    from denoising.util import getGPU
    from denoising.models.cnns.regularizers.gradient_test import GradientTest
    import numpy as np
    getGPU()
    test = GradientTest()
    config = {'grayscale': True,
              'filters': 24,
              'kernel_size': 3}
    operator = TNRDRegularizer(config)
    x = np.random.rand(2, 32, 32, 1)
    test.test_gradient(x, operator, ones_like=False)
