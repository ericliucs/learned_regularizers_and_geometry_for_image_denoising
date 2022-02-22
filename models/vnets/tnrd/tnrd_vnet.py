from models.vnets.vnet import VNet
from models.vnets.tnrd.tnrd import TNRD
from typing import Dict
from util import get_num_channels


class TNRDVNet(VNet):
    """Implementation of TNRD variational network.
    Code based on https://github.com/VLOGroup/denoising-variationalnetwork and
    https://github.com/VLOGroup/tdv/blob/master/model.py"""

    def __init__(self,
                 config: Dict,
                 **kwargs):
        """
        Initializes TNRD variational network

        Parameters
        ----------
        config: (Dict) - Config dictionary containing settings of network
        kwargs: kwargs of tf.keras.Model
        """

        super(TNRDVNet, self).__init__(S = config['S'],
                                       constant_dataterm_weight = False,
                                       constant_regularizer = False,
                                       use_prox = False,
                                       **kwargs)
        self.R = self.get_regularizer()

    def build_regularizer(self, stage: int = None):
        """Implements regularizer"""
        return TNRD(num_channels=get_num_channels(self.config),
                     filters=self.config['filters'],
                      kernel_size = self.config['kernel_size'],
                      name = f'Regularizer_{stage}')

    def get_config(self):
        """Return config for layer"""
        config = {'config': self.config}
        base_config = super(TNRDVNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))