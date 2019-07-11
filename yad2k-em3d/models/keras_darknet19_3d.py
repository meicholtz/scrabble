"""3D Darknet19 model defined in Keras."""
import functools
from functools import partial

from keras.layers import Conv3D, MaxPooling3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

import utils

# Partial wrapper for Convolution3D with static default argument
_DarknetConv3D = partial(Conv3D, padding='same')


@functools.wraps(Conv3D)
def DarknetConv3D(*args, **kwargs):
    """Wrapper to set Darknet weight regularizer for Convolution3D"""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs.update(kwargs)
    return _DarknetConv3D(*args, **darknet_conv_kwargs)


def DarknetConv3D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution3D followed by BatchNormalization and LeakyReLU"""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return utils.compose(
        DarknetConv3D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def bottleneck_3d_block(outer_filters, bottleneck_filters):
    """Bottleneck block of 3x3, 1x1, 3x3 convolutions"""
    return utils.compose(
        DarknetConv3D_BN_Leaky(outer_filters, (3, 3, 3)),
        DarknetConv3D_BN_Leaky(bottleneck_filters, (1, 1, 1)),
        DarknetConv3D_BN_Leaky(outer_filters, (3, 3, 3)))


def bottleneck_3d_x2_block(outer_filters, bottleneck_filters):
    """Bottleneck block of 3x3, 1x1, 3x3, 1x1, 3x3 convolutions"""
    return utils.compose(
        bottleneck_3d_block(outer_filters, bottleneck_filters),
        DarknetConv3D_BN_Leaky(bottleneck_filters, (1, 1, 1)),
        DarknetConv3D_BN_Leaky(outer_filters, (3, 3, 3)))


def darknet_body_3d():
    """Generate first 18 conv layers of Darknet-19"""
    return utils.compose(
        DarknetConv3D_BN_Leaky(16, (3, 3, 3)),
        MaxPooling3D(),
        DarknetConv3D_BN_Leaky(32, (3, 3, 3)),
        MaxPooling3D(),
        bottleneck_3d_block(64, 32),
        MaxPooling3D(),
        bottleneck_3d_block(128, 64),
        MaxPooling3D(),
        bottleneck_3d_x2_block(256, 128),
        MaxPooling3D(),
        bottleneck_3d_x2_block(512, 256))


def darknet19_3d(inputs):
    """Generate Darknet-19 model for Imagenet classification"""
    body = darknet_body_3d()(inputs)
    logits = DarknetConv3D(1000, (1, 1, 1), activation='softmax')(body)
    return Model(inputs, logits)
