from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.initializers import Constant
import numpy as np
from optotf.pad import pad2d, pad2d_transpose


def zero_mean_norm_ball(x, zero_mean=True, normalize=True, norm_bound=1.0, norm='l2', mask=None):
    """ Code from https://github.com/VLOGroup/denoising-variationalnetwork
    project onto zero-mean and norm-one ball
    :param x: tf variable which should be projected
    :param zero_mean: boolean True for zero-mean. default=True
    :param normalize: boolean True for l_2-norm ball projection. default:True
    :param norm_bound: defines the size of the norm ball
    :param norm: type of the norm
    :param mask: binary mask to compute the mean and norm
    :param axis: defines the axis for the reduction (mean and norm)
    :return: projection ops
    """

    mask = tf.ones(x.shape, dtype=np.float32)

    x_masked = x * mask

    if zero_mean:
        x_mean = tf.math.reduce_mean(x, axis=[0,1,2], keepdims = True)
        x_zm = x_masked - x_mean
    else:
        x_zm = x_masked

    if normalize:
        if norm == 'l2':
            x_proj = x_zm / tf.maximum(tf.sqrt(tf.reduce_sum(x_zm**2, axis=[0,1,2], keepdims=True)) /
                                                    norm_bound, 1)
        else:
            raise ValueError("Norm '%s' not defined." % norm)
    elif zero_mean:
        x_proj = x_zm
    else:
        x_proj = x

    return x_proj


class ZeroMeanNormBall(tf.keras.constraints.Constraint):
    """Keras constrain class for applying zero mean and projection onto norm ball"""
    def __call__(self, W):
        return zero_mean_norm_ball(W)


class ZeroMean(tf.keras.constraints.Constraint):
    """Keras constrain class for applying zero mean"""
    def __call__(self, W):
        return zero_mean_norm_ball(W, normalize=False)


class NormBall(tf.keras.constraints.Constraint):
    """Keras constrain class for applying projection onto norm ball"""
    def __call__(self, W):
        return zero_mean_norm_ball(W ,zero_mean=False)


class ForwardConv2d(Layer):
    """Implements forward operation of Conv2d operation from https://github.com/VLOGroup/tdv/blob/master/ddr/conv.py"""
    def __init__(self,
                 in_channels = 1,
                 stride=1,
                 dilation=1,
                 kernel_size=3,
                 pad=True,
                 **kwargs):
        """Initializes for convolution layer

        Parameters
        ----------
        in_channels: (int) - Number of in channels
        stride: (int) - Stride of convolution
        dilation: (int) - Dilation of convolution
        kernel_size: (int) - Kernel size
        pad: (bool) - Pads image with symmetric padding if True
        kwargs: kwargs of Keras layer
        """
        super(ForwardConv2d, self).__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.pad = pad
        super(ForwardConv2d, self).__init__(**kwargs)

    def call(self, inputs, training=None, mask=None):
        """Pads image and applies convolution to input"""
        x = inputs[0]
        weight = inputs[1]
        pad = weight.shape[1]//2
        if self.pad and pad > 0:
            x = pad2d(x, (pad, pad, pad, pad), mode='symmetric')
        x = tf.nn.conv2d(input=x,
                         filters=weight,
                         strides=self.stride,
                         padding='VALID',
                         dilations=self.dilation)
        return x

    def get_config(self):
        """Return config for layer"""
        config = {
            'in_channels': self.in_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'pad': self.pad}
        base_config = super(ForwardConv2d, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BackwardConv2d(ForwardConv2d):
    """Implements backward operation of Conv2d operation from https://github.com/VLOGroup/tdv/blob/master/ddr/conv.py"""

    def call(self, inputs, training=None, mask=None):
        """Applies conv transpose to input and then padding"""
        x = inputs[0]
        weight = inputs[1]
        output_s = inputs[2]

        inputs_shape = tf.shape(x)
        weight_shape = tf.shape(weight)

        # determine the output padding
        if output_s is None:
            H = (inputs_shape[1] - 1) * self.stride + weight_shape[0]
            W = (inputs_shape[2] - 1) * self.stride + weight_shape[0]
            output_shape = (inputs_shape[0], H, W, self.in_channels)
        else:
            output_shape = tf.shape(output_s) + [0,weight_shape[0]-1,weight_shape[0]-1,0]

        x = tf.nn.conv2d_transpose(input=x,
                                  filters=weight,
                                  output_shape=output_shape,
                                  strides=self.stride,
                                  dilations=self.dilation,
                                  padding='VALID')

        pad = weight.shape[1]//2
        if self.pad and pad > 0:
            x = pad2d_transpose(x, (pad, pad, pad, pad), mode='symmetric')
        return x


class Conv2d(tf.keras.Model):
    """Implements Conv2d operation from https://github.com/VLOGroup/tdv/blob/master/ddr/conv.py"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 invariant: bool = False,
                 stride: int = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 zero_mean: bool = False,
                 bound_norm: bool = False,
                 pad: bool = True,
                 **kwargs):
        """Initializes convolution operator model

       Parameters
       ----------
       in_channels: (int) - Number of in channels
       out_channels: (int) - Number of feature channels
       kernel_size: (int) - Kernel size
       invariant: (bool) - If True, conv kernels are invariant. Currently, not implemented
       stride: (int) - Stride of convolution
       dilation: (int) - Dilation of convolution
       bias: (bool) - Add bias to convolutions. Currently, not implemented
       zero_mean: (bool) - Constrain kernels to have zero mean
       bound_norm: (bool) - If True, bounds the norm of convolution kernels after each update step
       pad: (bool) - Pads image with symmetric padding if True
       kwargs: kwargs of tf.Keras.Model
        """
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.invariant = invariant
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        self.padding = 0
        self.pad = pad
        self._forward = ForwardConv2d(in_channels = in_channels,
                                      stride=stride,
                                      dilation=dilation,
                                      kernel_size=kernel_size,
                                      pad=pad)
        self._backward = BackwardConv2d(in_channels = in_channels,
                                        stride=stride,
                                        dilation=dilation,
                                        kernel_size=kernel_size,
                                        pad=pad)
        self._build()
        super(Conv2d, self).__init__(**kwargs)

    def _build(self):
        """Builds the convolution kernel weights"""

        if self.invariant:
            raise NotImplementedError
        if self.bias:
            raise NotImplementedError

        if self.zero_mean and self.bound_norm:
            constraint = ZeroMeanNormBall()
        elif self.zero_mean:
            constraint = ZeroMean()
        elif self.bound_norm:
            constraint = NormBall()
        else:
            constraint = None

        np_weights = np.random.normal(
            size=(self.kernel_size, self.kernel_size, self.in_channels, self.out_channels),
            scale=np.sqrt(1 / np.prod(self.in_channels * self.kernel_size ** 2)))

        self.weight = self.add_weight(name='conv_weight',
                             shape=[self.kernel_size, self.kernel_size, self.in_channels, self.out_channels],
                             initializer=Constant(np_weights), trainable=True,
                            constraint=constraint)

    def get_weight(self):
        """Grabs weight for conv operation. Also applied any pre-processing to weights"""
        if self.invariant:
            raise NotImplementedError
        else:
            return self.weight

    def forward(self, x):
        """Apply forward convolution operation"""
        weight = self.get_weight()
        return self._forward([x, weight])

    def backward(self, x, output_s = None):
        """Apply backward convolution operation"""
        weight = self.get_weight()
        return self._backward([x, weight, output_s])

    def get_config(self):
        """Retrieve config of model"""
        config = {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'invariant': self.invariant,
            'stride': self.stride,
            'dilation': self.dilation,
            'bias': self.bias,
            'zero_mean': self.zero_mean,
            'bound_norm': self.bound_norm,
            'pad': self.pad,
            'padding': self.padding}

        base_config = super(Conv2d, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvScale2d(Conv2d):
    """Implements ConvScale2d operation from https://github.com/VLOGroup/tdv/blob/master/ddr/conv.py"""
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 invariant: bool = False,
                 stride: int = 2,
                 bias: bool = False,
                 zero_mean: bool = False,
                 bound_norm: bool = False,
                 **kwargs):
        """Initializes convolution operator model
           Parameters
           ----------
           in_channels: (int) - Number of in channels
           out_channels: (int) - Number of feature channels
           kernel_size: (int) - Kernel size
           invariant: (bool) - If True, conv kernels are invariant. Currently, not implemented
           stride: (int) - Stride of convolution
           dilation: (int) - Dilation of convolution
           bias: (bool) - Add bias to convolutions. Currently, not implemented
           zero_mean: (bool) - Constrain kernels to have zero mean
           bound_norm: (bool) - If True, bounds the norm of convolution kernels after each update step
           pad: (bool) - Pads image with symmetric padding if True
           kwargs: kwargs of tf.Keras.Model
        """

        super(ConvScale2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            invariant=invariant, stride=stride, dilation=1, bias=bias,
            zero_mean=zero_mean, bound_norm=bound_norm, **kwargs)

        # create the convolution kernel
        if self.stride > 1:
            np_k = np.asarray([1, 4, 6, 4, 1], dtype=np.float32)[:, np.newaxis]
            np_k = np_k @ np_k.T
            np_k /= np_k.sum()
            np_k = np.reshape(np_k, (5, 5, 1, 1))
            self.blur = tf.constant(np_k, dtype=tf.float32)

    def get_weight(self):
        """Apply convolution kernel blurring"""
        weight = super().get_weight()
        if self.stride > 1:
            weight = tf.reshape(weight, [self.kernel_size, self.kernel_size, 1, -1])
            weight = tf.transpose(weight, perm=[3, 0, 1, 2])
            for i in range(self.stride // 2):
                weight = tf.nn.conv2d(input=weight, filters=self.blur, strides=[1, 1, 1, 1], padding=[[0,0], [4,4], [4,4], [0,0]])
            weight = tf.transpose(weight, perm=[1, 2, 3, 0])
            weight = tf.reshape(weight, [self.kernel_size + 2 * self.stride, self.kernel_size + 2 * self.stride, self.in_channels, self.out_channels])
        return weight


class ConvScaleTranspose2d(ConvScale2d):
    """Implements ConvScaleTranspose2d operation from https://github.com/VLOGroup/tdv/blob/master/ddr/conv.py"""

    def forward(self, x, output_s = None):
        """Change forward operation to conv transpose"""
        weight = self.get_weight()
        return self._backward([x, weight, output_s])

    def backward(self, x):
        """Change backward operation to convolution"""
        weight = self.get_weight()
        return self._forward([x, weight])


if __name__ == "__main__":
    """For Testing"""
    from util import getGPU
    from models.cnns.regularizers.gradient_test import GradientTest
    getGPU()
    test = GradientTest()

    x = np.random.rand(2, 64, 64, 1)
    operator = Conv2d(1, 4, kernel_size=5, name='Conv2d', stride=2)
    test.test_gradient(x, operator)

    x = np.random.rand(2, 64, 64, 1)
    operator = ConvScale2d(1, 4, kernel_size=5, name='ConvScale2d', stride=2)
    test.test_gradient(x, operator)

    x = np.random.rand(2, 64, 64, 2)
    operator = ConvScaleTranspose2d(2, 2, kernel_size=5, name='ConvScaleTranspose2d', stride=2)
    test.test_gradient(x, operator)
