import tensorflow as tf
from denoising.models.cnns.regularizers.regularizer import Regularizer
from typing import List
import numpy as np


def convert_filter_to_tensor(filter: List):
    """Converts 2d list filter to 4d numpy filter
    """
    filter = np.array(filter)[:,:,np.newaxis, np.newaxis]
    return tf.constant(filter, dtype=tf.float32)


class TVRegularizer(Regularizer):
    """Implements Total Variation regularizer as detailed in
    https://www.sciencedirect.com/science/article/abs/pii/016727899290242F?via%3Dihub
    by Rudin, Osher, Fatemi"""

    def __init__(self,
                 config,
                 **kwargs):
        """Initializes the regularizer

        Parameters
        ----------
        config: (Dict) - Dictionary of configuration settings for regularizer
        """
        super(TVRegularizer, self).__init__()

        # Initialize kernels
        self.dx_forward = convert_filter_to_tensor([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
        self.dy_forward = convert_filter_to_tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
        self.dx_backward = convert_filter_to_tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
        self.dy_backward = convert_filter_to_tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
        self.dy_stepback_forward = convert_filter_to_tensor([[0, -1, 1], [0, 0, 0], [0, 0, 0]])
        self.dx_stepleft_forward = convert_filter_to_tensor([[0, 0, 0], [-1, 0, 0], [1, 0, 0]])
        self.eps = tf.constant(float(1e-2), dtype=tf.float32)

    def forward(self, x):
        """Computes forward operation of Total Variation.
        """
        x_p = tf.pad(x, [(0,0), (1,1), (1,1), (0,0)], mode='SYMMETRIC')
        dx = tf.nn.conv2d(input=x_p, filters=self.dx_forward, strides=[1, 1, 1, 1], padding='VALID')
        dy = tf.nn.conv2d(input=x_p, filters=self.dy_forward, strides=[1, 1, 1, 1], padding='VALID')
        tv = tf.sqrt(tf.square(dx) + tf.square(dy) + self.eps)
        return tv

    def backward(self, x):

        # Pad the image with symmetric padding. i.e. if we are padding a boundary, just copy that boundary and use it
        # as the padding
        u_p = tf.pad(tensor=x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')

        # Apply convolutions
        dxforwardu = tf.nn.conv2d(input=u_p, filters=self.dx_forward, strides=[1, 1, 1, 1], padding='VALID')
        dxbacku = tf.nn.conv2d(input=u_p, filters=self.dx_backward, strides=[1, 1, 1, 1], padding='VALID')
        dyforwardu = tf.nn.conv2d(input=u_p, filters=self.dy_forward, strides=[1, 1, 1, 1], padding='VALID')
        dybacku = tf.nn.conv2d(input=u_p, filters=self.dy_backward, strides=[1, 1, 1, 1], padding='VALID')
        dystepbackforwardu = tf.nn.conv2d(input=u_p, filters=self.dy_stepback_forward, strides=[1, 1, 1, 1],
                                          padding='VALID')
        dxstepleftforwardu = tf.nn.conv2d(input=u_p, filters=self.dx_stepleft_forward, strides=[1, 1, 1, 1],
                                          padding='VALID')

        # Compute respective gradients at different locations
        F = tf.sqrt(tf.square(dxforwardu) + tf.square(dyforwardu) + self.eps)
        G = tf.sqrt(tf.square(dxbacku) + tf.square(dystepbackforwardu) + self.eps)
        H = tf.sqrt(tf.square(dxstepleftforwardu) + tf.square(dybacku) + self.eps)

        # Scale each piece by the norms of the respective gradients and add
        Curvature = tf.divide(dxforwardu, F) - tf.divide(dxbacku, G) + tf.divide(dyforwardu, F) - tf.divide(dybacku, H)
        return Curvature

    def energy(self, x):
        """Compute energy of forward operation"""
        raise NotImplementedError
        #return self.forward(x)

    def grad(self, x, get_energy=False):
        if get_energy:
            raise NotImplementedError
        grad = self.backward(x)
        return grad


if __name__ == "__main__":
    """For Testing"""
    from denoising.util import getGPU
    from denoising.models.cnns.regularizers.gradient_test import GradientTest
    import numpy as np
    getGPU()
    test = GradientTest()
    operator = TVRegularizer({})
    x = np.random.randn(2, 64, 64, 1).astype(np.float32)
    test.test_gradient(x, operator)