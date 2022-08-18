from denoising.models.cnns.regularizers.tdv.conv import Conv2d, ConvScale2d, ConvScaleTranspose2d
import tensorflow as tf
from tensorflow.keras.layers import Layer


class StudentT2(Layer):
    """Implements log-student-t-distribution operation. Code based on
    https://github.com/VLOGroup/tdv/blob/master/ddr/tdv.py"""

    def __init__(self, alpha=1, **kwargs):
        """Initalize alpha"""
        super(StudentT2, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs, training=None, mask=None):
        """Call operation on input"""
        x = inputs
        d = 1 + self.alpha * x ** 2
        return tf.math.log(d)/(2*self.alpha), x/d

    def get_config(self):
        """Return config for layer"""
        config = {'alpha': self.alpha}
        base_config = super(StudentT2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MicroBlock(tf.keras.Model):
    """Implements MicroBlock operation. Code based on https://github.com/VLOGroup/tdv/blob/master/ddr/tdv.py"""
    def __init__(self,
                 num_features: int,
                 bound_norm: bool=False,
                 invariant: bool=False,
                 **kwargs):
        """Initializes microblock

        Parameters
        ----------
        num_features: (int) - Number of features computed by convolutions
        bound_norm: (bool) - If True, bounds the norm of convolution kernels after each update step
        invariant: (bool) - If True, conv kernels are invariant. Currently, not implemented
        kwargs: kwargs of tf.keras.Model
        """
        self.num_features = num_features
        self.bound_norm = bound_norm
        self.invariant = invariant
        super(MicroBlock, self).__init__(**kwargs)
        self.conv1 = Conv2d(num_features, num_features, kernel_size=3, invariant=invariant,
                              bound_norm=bound_norm, bias=False, name=f'{self.name}_tdv_conv2d_1')
        self.act = StudentT2(alpha=1)
        self.conv2 = Conv2d(num_features, num_features, kernel_size=3, invariant=invariant,
                               bound_norm=bound_norm, bias=False, name=f'{self.name}_tdv_conv2d_2')
        self.act_prime = None

    def forward(self, x):
        """Forward pass of Microblock"""
        a, ap = self.act(self.conv1.forward(x))
        self.act_prime = ap
        x = x + self.conv2.forward(a)
        return x

    def backward(self, grad_out):
        """Backward pass of Microblock"""
        assert not self.act_prime is None
        out = grad_out + self.conv1.backward(self.act_prime*self.conv2.backward(grad_out))
        return out

    def get_config(self):
        """Return config for model"""
        config = {'num_features': self.num_features,
                  'bound_norm': self.bound_norm,
                  'invariant': self.invariant}
        base_config = super(MicroBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MacroBlock(tf.keras.Model):
    '''Defines a TDV Macroblock as detailed in https://arxiv.org/abs/2001.05005.
    Code based on https://github.com/VLOGroup/tdv.'''

    def __init__(self,
                 num_features: int,
                 num_scales: int = 3,
                 multiplier:  int = 1,
                 bound_norm: bool = False,
                 invariant: bool = False,
                 **kwargs):
        """Initialization of the MacroBlock

        Parameters
        ----------
        num_features: int
            Number of features within MacroBlock
        num_scales: int
            Number of scales of MacroBlock. Defines number of MicroBlocks.
        multiplier: int
            Multiplier for features of MicroBlocks.
        bound_norm: bool
            If True, bound norm of convolution kernels.
        invariant: bool
            If True, apply invariant processing to convolution kernels. Currently not implemented
        """
        super(MacroBlock, self).__init__()
        self.num_features = num_features
        self.num_scales = num_scales
        self.multiplier = multiplier
        self.bound_norm = bound_norm
        self.invariant = invariant

        # micro blocks
        self.mb = []
        for i in range(num_scales - 1):
            b = [
                MicroBlock(num_features * multiplier ** i, name=f'{self.name}_Microblock_{i}_1', bound_norm=bound_norm, invariant=invariant),
                MicroBlock(num_features * multiplier ** i, name=f'{self.name}_Microblock_{i}_2', bound_norm=bound_norm, invariant=invariant)]
            self.mb.append(b)
        # the coarsest scale has only one microblock
        self.mb.append([
            MicroBlock(num_features * multiplier ** (num_scales - 1), name=f'{self.name}_Microblock_{num_scales}_1', bound_norm=bound_norm, invariant=invariant)
        ])
        self.mb = list(self.mb)

        # down/up sample
        self.conv_down = []
        self.conv_up = []
        for i in range(1, num_scales):
            self.conv_down.append(
                ConvScale2d(num_features * multiplier ** (i - 1), num_features * multiplier ** i, kernel_size=3,
                            bias=False, invariant=invariant, bound_norm=bound_norm, name=f'{self.name}_conv_down_{i}')
            )
            self.conv_up.append(
                ConvScaleTranspose2d(num_features * multiplier ** (i - 1), num_features * multiplier ** i,
                                     kernel_size=3, bias=False, invariant=invariant, bound_norm=bound_norm, name=f'{self.name}_conv_up_{i}')
            )
        super(MacroBlock, self).__init__(**kwargs)

    def get_config(self):
        """Return config for model"""
        config = {'num_features': self.num_features,
                  'num_scales': self.num_scales,
                  'multiplier': self.multiplier,
                  'bound_norm': self.bound_norm,
                  'invariant': self.invariant}
        base_config = super(MacroBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def forward(self, x):
        """Apply MacroBlock to list of input tensors
        """
        assert len(x) == self.num_scales

        # down scale and feature extraction
        for i in range(self.num_scales - 1):
            # 1st micro block of scale
            x[i]  = self.mb[i][0].forward(x[i])
            # down sample for the next scale
            x_i_down = self.conv_down[i].forward(x[i])
            if x[i + 1] is None:
                x[i + 1] = x_i_down
            else:
                x[i + 1] = x[i + 1] + x_i_down

        # on the coarsest scale we only have one micro block
        x[self.num_scales - 1] = self.mb[self.num_scales - 1][0].forward(x[self.num_scales - 1])

        # up scale the features
        for i in range(self.num_scales - 1)[::-1]:
            # first upsample the next coarsest scale
            x_ip1_up = self.conv_up[i].forward(x[i + 1], output_s = x[i])
            # skip connection
            x[i] = x[i] + x_ip1_up
            # 2nd micro block of scale
            x[i] = self.mb[i][1].forward(x[i])

        return x

    def backward(self, x):
        """Apply backward operation to of macroblock to list of input tensors"""

        # backward of up scale the features
        for i in range(self.num_scales - 1):
            # 2nd micro block of scale
            x[i] = self.mb[i][1].backward(x[i])
            # first upsample the next coarsest scale
            x_ip1_up = self.conv_up[i].backward(x[i])
            # skip connection
            if x[i + 1] is None:
                x[i + 1] = x_ip1_up
            else:
                x[i + 1] = x[i + 1] + x_ip1_up

        # on the coarsest scale we only have one micro block
        x[self.num_scales - 1] = self.mb[self.num_scales - 1][0].backward(x[self.num_scales - 1])

        # down scale and feature extraction
        for i in range(self.num_scales - 1)[::-1]:
            # down sample for the next scale
            x_i_down = self.conv_down[i].backward(x[i + 1], output_s = x[i])
            x[i] = x[i] + x_i_down
            # 1st micro block of scale
            x[i] = self.mb[i][0].backward(x[i])

        return x


if __name__ == "__main__":
    """For Testing"""
    from denoising.util import getGPU
    from denoising.models.cnns.regularizers.gradient_test import GradientTest
    import numpy as np
    getGPU()
    test = GradientTest()
    operator = MicroBlock(num_features=2, name='MicroBlock')
    x = np.random.rand(2, 32, 32, 2)
    test.test_gradient(x, operator)
    x = np.random.rand(2, 32, 32, 2)
    operator = MacroBlock(num_features=2, name='MacroBlock')
    test.test_gradient(x, operator, num_scales=3)
