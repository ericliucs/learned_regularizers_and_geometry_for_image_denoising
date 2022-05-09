import tensorflow as tf
from abc import ABC, abstractmethod


class Regularizer(tf.keras.Model, ABC):
    """Abstract class for regularizer model"""

    @abstractmethod
    def forward(self, x):
        """Computes forward operation of regularizer"""
        raise NotImplementedError

    @abstractmethod
    def backward(self, x):
        """Computes backward operation of regularizer i.e. gradient of forward operation"""
        raise NotImplementedError

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
        """Compute energy/gradient of regularizer"""
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

    def call(self, inputs, training=None, mask=None):
        return self.grad(inputs)