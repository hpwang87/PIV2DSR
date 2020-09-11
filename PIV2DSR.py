# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:48:53 2019

@author: H.P. Wang
github:  https://github.com/hpwang
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

"""
GPU device
"""
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def res_block(input_tensor, filters, scale=0.2, gamma=0):
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', 
               kernel_regularizer=regularizers.l2(gamma))(input_tensor)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
               kernel_regularizer=regularizers.l2(gamma))(x)
    
    if scale:
        """
        Lambda function: x = x*scale
        """
        x = Lambda(lambda t: t * scale)(x)
        
    x = Add()([x, input_tensor])

    return x


def sub_pixel_Conv2D(scale=2, **kwargs):
    ''' 
    tf.depth_to_space for tensorflow1.x
    tf.nn.depth_to_space for tensorflow2.x
    Lambda(function, output_shape=None, mask=None, arguments=None, **kwargs)
    '''
    return Lambda(lambda x: tf.nn.depth_to_space(x, scale), **kwargs)



def upsample(input_tensor, filters, gamma=0):
    """
    pixelshuffle: enlarge the channels (features) to upsample the resolution
    """
    x = Conv2D(filters=filters * 4, kernel_size=3, strides=1, padding='same',
               kernel_regularizer=regularizers.l2(gamma))(input_tensor)

    x = sub_pixel_Conv2D(scale=2)(x)
    x = Activation('relu')(x)
    return x



def generator(input_shape, filters=32, n_id_block=6, n_sub_block=2, gamma=0):
    inputs = Input(shape=input_shape)
        
    """
    first convolutional layer
    """
    x = x_0 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
                     kernel_regularizer=regularizers.l2(gamma))(inputs)
    """
    additonal convolutional layer
    """
    x_0 = Conv2D(filters=2*(2*n_sub_block)**2, kernel_size=3, strides=1, padding='same',
                 kernel_regularizer=regularizers.l2(gamma))(x_0)
    """
    second convolutional layer
    """
    x = x_1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
                     kernel_regularizer=regularizers.l2(gamma))(x)
    """
    third residual blocks x6
    """
    for _ in range(n_id_block):
        x = res_block(x, filters=filters)
    x = x_1 = Add()([x_1, x])
    """
    fourth convolutional layer
    """
    x = Conv2D(filters=2*(2*n_sub_block)**2, kernel_size=3, strides=1, padding='same',
               kernel_regularizer=regularizers.l2(gamma))(x)
    """
    add the features with x_0 and x
    """
    x = Add()([x_0, x])
    """
    enhance the resolution by 4 times
    """
    x = sub_pixel_Conv2D(scale=2*n_sub_block)(x)

    return Model(inputs=inputs, outputs=x)