from models.vnets.vnet import VNet
from models.vnets.tdv.tdv import TDV
from typing import Dict
from util import get_num_channels


class TDVNet(VNet):
    """Implementation of TDV variational network.
    Code based on https://github.com/VLOGroup/tdv/blob/master/model.py"""

    def __init__(self,
                 config: Dict,
                 **kwargs):
        """
        Initializes variational network

        Parameters
        ----------
        config: (Dict) - Config dictionary containing settings of network
        kwargs: kwargs of tf.keras.Model
        """

        super(TDVNet, self).__init__(S=config['S'],
                                     use_prox=config['use_prox'],
                                     checkpoint = config['checkpoint'],
                                     num_non_checkpoints=5)

        self.config = config
        self.R = self.get_regularizer()
        super(TDVNet, self).__init__(**kwargs)

    def build_regularizer(self, stage: int = None):
        """Must defines regularizer"""
        return TDV(num_channels=get_num_channels(self.config),
                     filters=self.config['filters'],
                     num_scales=self.config['num_scales'],
                     multiplier=self.config['multiplier'],
                     num_mb=self.config['num_mb'],
                     name='TDV')

    def get_config(self):
        """Return config for layer"""
        config = {'config': self.config}
        base_config = super(TDVNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

