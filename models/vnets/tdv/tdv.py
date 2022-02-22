import tensorflow as tf
from models.vnets.tdv.conv import Conv2d
from models.vnets.tdv.blocks import MacroBlock


class TDV(tf.keras.Model):
    """Implements TDV regularizer as detailed in https://arxiv.org/pdf/2001.05005.pdf
    Code based https://github.com/VLOGroup/tdv/blob/master/ddr/tdv.py"""

    def __init__(self,
                 num_channels: int = 1,
                 filters: int = 32,
                 num_scales: int = 3,
                 multiplier: int = 1,
                 num_mb: int = 3,
                 **kwargs):
        """Initializes the regularizer

        Parameters
        ----------
        num_channels: (int) - Number of channels of images
        filters: (int) - Number of features in model
        num_scales: (int) - Number of scales for macro and micro blocks
        multiplier: (int) - Multiplier for macro blocks
        num_mb: (int) - Number of macroblocks
        kwargs: (int) - tf.keras.Model kwargs
        """
        super(TDV, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.filters = filters
        self.num_scales = num_scales
        self.multiplier = multiplier
        self.num_mb = num_mb

        self.K1 = Conv2d(in_channels=num_channels, out_channels=filters,
                            kernel_size=3, zero_mean=True,
                            invariant=False, bound_norm=True, bias=False, name='K1')
        self.mb = [MacroBlock(num_features=filters, name=f'Macroblock_{_}',
                              num_scales=num_scales, bound_norm=False,
                              invariant=False, multiplier=multiplier)
                   for _ in range(num_mb)]
        self.KN = Conv2d(in_channels=filters, out_channels=num_channels,
                            kernel_size=1, invariant=False, bound_norm=False, bias=False, zero_mean=False, name='KN')

    def get_config(self):
        """Returns configuration of TDV regularizer"""
        config = {
            'in_channels': self.in_channels,
            'filters': self.filters,
            'num_scales': self.num_scales,
            'multiplier': self.multiplier,
            'num_mb': self.num_mb
        }
        base_config = super(TDV, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def forward(self, x):
        """Computes forward operation of TDV.
        """
        # extract features
        x = self.K1.forward(x)
        # apply mb
        x = [x, ] + [None for i in range(self.num_scales - 1)]
        for i in range(self.num_mb):
            x = self.mb[i].forward(x)
        # compute the output
        out = self.KN.forward(x[0])
        return out

    def backward(self, x):
        """Computes backward operation of TDV i.e. gradient of forward operation"""
        # compute the output
        x = self.KN.backward(x)
        # apply mb
        x = [x, ] + [None for i in range(self.num_scales - 1)]
        for i in range(self.num_mb)[::-1]:
            x = self.mb[i].backward(x)
        # extract features
        x = self.K1.backward(x[0])
        return x

    def _activation(self, x):
        """Scale tensor of ones by the number of features"""
        return tf.ones_like(x) / self.filters

    def _potential(self, x):
        """Scale by number of features"""
        return x / self.filters

    def energy(self, x):
        """Compute energy of forward operation"""
        x = self.forward(x)
        return self._potential(x)

    def grad(self, x, get_energy=False):
        """Compute energy/gradient of TDV model"""
        # compute the energy
        x = self.forward(x)
        if get_energy:
            energy = self._potential(x)
        # and its gradient
        x = self._activation(x)
        grad = self.backward(x)
        if get_energy:
            return energy, grad
        else:
            return grad


if __name__ == "__main__":
    """For Testing"""
    from util import getGPU
    from models.vnets.tdv.test import GradientTest
    import numpy as np
    getGPU()
    test = GradientTest()
    operator = TDV()
    x = np.random.rand(2, 32, 32, 1)
    test.test_gradient(x, operator)
