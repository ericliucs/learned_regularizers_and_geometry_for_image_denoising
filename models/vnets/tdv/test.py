import unittest
import tensorflow as tf
import numpy as np


class GradientTest(unittest.TestCase):
    """Unit test for testing forward and backwards computations of TDV operators
    Code based on unit test from https://github.com/VLOGroup/tdv/blob/master/ddr/tdv.py"""

    @staticmethod
    def _grad(x: np.ndarray, operator: tf.keras.Model, num_scales: int = None):
        """Computes forward operation on x and then uses backward operation to compute its gradient

        Parameters
        ----------
        x : (np.ndarray) - Input into model
        operator (tf.keras.Model): Instantiated Operator that has forwards and backwards methods.
        num_scales: (int) - Number of scales if operator is micro or macro block
        """
        if num_scales is not None:
            x = [x, ] + [None for i in range(num_scales-1)]
        energy = operator.forward(x)
        x = tf.ones_like(energy)
        if num_scales is not None:
            x = [x, ] + [None for i in range(num_scales-1)]
        grad = operator.backward(x)
        if isinstance(energy, list):
            energy = energy[0]
        if isinstance(grad, list):
            grad = grad[0]
        return energy, grad

    def test_gradient(self, x: np.ndarray, operator: tf.keras.Model, num_scales: int = None):
        """Tests forward and backward computations of TDV operator

        Parameters
        ----------
        x : (np.ndarray) - Input into model
        operator (tf.keras.Model): Instantiated Operator that has forwards and backwards methods.
        num_scales: (int) - Number of scales if operator is micro or macro block
        """
        # clear in case we are running multiple tests
        tf.keras.backend.clear_session()

        # compute the gradient using the implementation
        energy, grad = self._grad(x, operator, num_scales = num_scales)
        grad_scale = np.sum(grad)

        # check it numerically
        epsilon = 1e-3
        l_p = np.sum(self._grad(x + epsilon, operator, num_scales=num_scales)[0])
        l_n = np.sum(self._grad(x - epsilon, operator, num_scales=num_scales)[0])
        grad_scale_num = (l_p - l_n) / (2 * epsilon)

        condition = np.abs(grad_scale - grad_scale_num) < 1e-1
        print(f'{operator.name}: grad_scale: {grad_scale:.7f} num_grad_scale {grad_scale_num:.7f} success: {condition}')
        self.assertTrue(condition)
