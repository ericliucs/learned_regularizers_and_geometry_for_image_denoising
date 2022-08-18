import unittest
import tensorflow as tf
import numpy as np


class GradientTest(unittest.TestCase):
    """Unit test for testing forward and backwards computations of variational operators
    Code based on unit test from https://github.com/VLOGroup/tdv/blob/master/ddr/tdv.py"""

    @staticmethod
    def _grad(x: np.ndarray, operator: tf.keras.Model, num_scales: int = None, ones_like: bool = True):
        """Computes forward operation on x and then uses backward operation to compute its gradient

        Parameters
        ----------
        x : (np.ndarray) - Input into model
        operator (tf.keras.Model): Instantiated Operator that has forwards and backwards methods.
        num_scales: (int) - Number of scales if operator is micro or macro block
        ones_like: (bool) - Determines if backward input is x or tf.ones_like(x)
        """
        if num_scales is not None:
            x = [x, ] + [None for i in range(num_scales-1)]
        energy = operator.forward(x)
        if isinstance(energy, list):
            energy = energy[0]
        if ones_like:
            x = tf.ones_like(energy)
        if num_scales is not None:
            x = [x, ] + [None for i in range(num_scales-1)]
        grad = operator.backward(x)
        if isinstance(grad, list):
            grad = grad[0]
        return energy, grad

    def test_gradient(self, x: np.ndarray,
                      operator: tf.keras.Model,
                      num_scales: int = None,
                      ones_like: bool = True):
        """Tests forward and backward computations of TDV operator

        Parameters
        ----------
        x : (np.ndarray) - Input into model
        operator (tf.keras.Model): Instantiated Operator that has forwards and backwards methods.
        num_scales: (int) - Number of scales if operator is micro or macro block
        ones_like: (bool) - Determines if backward input is x or tf.ones_like(x)
        """

        # clear in case we are running multiple tests
        tf.keras.backend.clear_session()

        # Use tensorflow to automatically compute gradients
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(x)
            energy, grad_compute = self._grad(x, operator, num_scales=num_scales, ones_like=ones_like)
        grad_auto = g.gradient(energy, x)

        # Check auto gradient against manually computed gradient
        grad_compute_total = np.sum(grad_compute)
        grad_auto_total = np.sum(grad_auto)
        condition = np.abs(grad_compute_total - grad_auto_total) < 1e-3
        print(f'{operator.name}: grad_compute: {grad_compute_total:.7f} grad_auto {grad_auto_total:.7f} success: {condition}')
        self.assertTrue(condition)
