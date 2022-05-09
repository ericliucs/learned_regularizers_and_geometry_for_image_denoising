from models.cnns.regularizers.tdv.conv import Conv2d
from models.cnns.regularizers.tdv.blocks import MacroBlock
from models.cnns.regularizers.regularizer import Regularizer
from util import get_num_channels
from typing import Dict


class TDVRegularizer(Regularizer):
    """Implements TDV regularizer as detailed in https://arxiv.org/pdf/2001.05005.pdf
    Code based https://github.com/VLOGroup/tdv/blob/master/ddr/tdv.py"""

    def __init__(self,
                 config: Dict,
                 **kwargs):
        """Initializes the regularizer

        Parameters
        ----------
        config: (Dict) - Dictionary containing TDV configuration
        """
        super(TDVRegularizer, self).__init__()
        self.num_channels = get_num_channels(config)
        self.filters = config['filters']
        self.num_scales = config['num_scales']
        self.multiplier = config['multiplier']
        self.num_mb = config['num_mb']

        self.K1 = Conv2d(in_channels=self.num_channels, out_channels=self.filters,
                            kernel_size=3, zero_mean=True,
                            invariant=False, bound_norm=True, bias=False, name='K1')
        self.mb = [MacroBlock(num_features=self.filters, name=f'Macroblock_{_}',
                              num_scales=self.num_scales, bound_norm=False,
                              invariant=False, multiplier=self.multiplier)
                   for _ in range(self.num_mb)]
        self.KN = Conv2d(in_channels=self.filters, out_channels=self.num_channels,
                            kernel_size=1, invariant=False, bound_norm=False, bias=False, zero_mean=False, name='KN')
        super(TDVRegularizer, self).__init__(**kwargs)

    def get_config(self):
        """Returns configuration of TDV regularizer"""
        config = {
            'in_channels': self.num_channels,
            'filters': self.filters,
            'num_scales': self.num_scales,
            'multiplier': self.multiplier,
            'num_mb': self.num_mb
        }
        base_config = super(TDVRegularizer, self).get_config()
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


if __name__ == "__main__":
    """For Testing"""
    from util import getGPU
    from models.cnns.regularizers.gradient_test import GradientTest
    import numpy as np
    getGPU()
    test = GradientTest()
    operator = TDVRegularizer({
        'filters': 32,
        'num_scales': 3,
        'multiplier': 1,
        'num_mb': 3,
        'grayscale': True,
    })
    x = np.random.rand(2, 32, 32, 1)
    test.test_gradient(x, operator)