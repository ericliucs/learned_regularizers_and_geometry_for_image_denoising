from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.constraints import MinMaxNorm
import numpy as np


class L2DenoiseDataterm:
    """Implements L2DenoiseDataTerm for Variational Network.
    Code based on https://github.com/VLOGroup/tdv/blob/master/model.py"""

    @staticmethod
    def energy(x, z):
        """Compute the energy"""
        return 0.5 * (x - z) ** 2

    @staticmethod
    def prox(x, z, tau):
        """Compute proximal dataterm"""
        return (x + tau * z) / (1 + tau)

    @staticmethod
    def grad( x, z):
        """Compute grad of energy"""
        return x - z


class VNet(tf.keras.Model):
    """Implementation of denoising variational network.
    Code based on https://github.com/VLOGroup/tdv/blob/master/model.py"""
    def __init__(self,
                 S: int = 10,
                 constant_dataterm_weight: bool = True,
                 constant_regularizer: bool = True,
                 use_prox: bool=False,
                 checkpoint: bool=False,
                 num_non_checkpoints: int = 0,
                 **kwargs):
        """
        Initializes variational network

        Parameters
        ----------
        S: (int) - Number of stages of variational network
        constant_dataterm_weight: (int) - If True, make lmbda dataterm weight for each stage like in TNRD model
        constant_regularizer: (int) - If True, allow for regularizer weights to change at every stage
        use_prox: (bool) - If True, use proximal descent step
        kwargs
        """
        super(VNet, self).__init__(**kwargs)
        self.S = S
        self.constant_dataterm_weight = constant_dataterm_weight
        self.constant_regularizer = constant_regularizer
        self.use_prox = use_prox
        self.checkpoint = checkpoint
        self.num_non_checkpoints = num_non_checkpoints
        self.checkpoint_after = self.S - num_non_checkpoints

        # Get weights for descent step and dataterm
        if constant_dataterm_weight:
            self.T = self.add_weight(name='T',
                                     shape=[1],
                                     initializer=Constant(np.asarray(0.001)),
                                     constraint=MinMaxNorm(min_value=0, max_value=1000),
                                     trainable=True)
            self.lmbda = self.add_weight(name='lmbda',
                                         shape=[1],
                                         initializer=Constant(np.asarray(0.01))
                                         , constraint=MinMaxNorm(min_value=0, max_value=1000), trainable=True)
        else:
            self.lmbda = [self.add_weight(name=f'lmbda_{i}',
                                          shape=[1],
                                          initializer=Constant(np.asarray(0.01))
                                          , constraint=MinMaxNorm(min_value=0, max_value=1000), trainable=True)
                          for i in range(S)]
        # Get dataterm
        self.D = L2DenoiseDataterm()

    def get_regularizer(self):
        """Grabs regularizer(s)"""
        if self.constant_regularizer:
            self.R = self.build_regularizer()
        else:
            self.R = [self.build_regularizer(i+1) for i in range(self.S)]
        return self.R

    @abstractmethod
    def build_regularizer(self, stage: int = None):
        """Must define regularizer"""
        raise NotImplementedError

    def call(self, inputs, training=None, mask=None):
        """Compute variational network"""
        x = inputs
        z = x

        if self.constant_dataterm_weight:
            tau = self.T / self.S

        grad = tf.recompute_grad(self.R.grad)
        for s in range(1, self.S + 1):
            if self.checkpoint:
                if s > self.checkpoint_after:
                    grad_R = self.R.grad(x)
                else:
                    grad_R = grad(x)

        for s in range(1, self.S + 1):

            if self.constant_regularizer:
                grad_fnc = self.R.grad
            else:
                grad_fnc = self.R[s - 1].grad

            if self.checkpoint:
                if s > self.checkpoint_after:
                    grad_R = grad_fnc(x)
                else:
                    grad_R = tf.recompute_grad(grad_fnc)(x)
            else:
                grad_R = grad_fnc(x)

            if self.constant_dataterm_weight:
                if self.use_prox:
                    x = self.D.prox(x - tau * grad_R, z, self.lmbda / self.S)
                else:
                    x = x - tau * grad_R - self.lmbda / self.S * self.D.grad(x, z)
            else:
                x = x - grad_R - self.lmbda[s - 1] * self.D.grad(x, z)

        return x

    def get_config(self):
        """Return config for model"""
        config = {'S': self.S,
                  'constant_dataterm_weight': self.constant_dataterm_weight,
                  'constant_regularizer': self.constant_regularizer,
                  'use_prox': self.use_prox}
        base_config = super(VNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))