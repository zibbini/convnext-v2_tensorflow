import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras
from keras import utils

from .convnext_v1 import DropPath, DownSample, Stem, Head


class GRN(layers.Layer):
    def __init__(self, dim, name='grn', **kwargs):
        super().__init__(**kwargs)
        self.gamma = tf.Variable(
            initial_value=tf.zeros((1, 1, 1, dim), dtype=self.compute_dtype),
            trainable=True,
            dtype=self.compute_dtype,
            name=f'{name}/gamma'
        )
        self.beta = tf.Variable(
            initial_value=tf.zeros((1, 1, 1, dim), dtype=self.compute_dtype),
            trainable=True,
            dtype=self.compute_dtype,
            name=f'{name}/beta'
        )

    def call(self, x):
        Gx = tf.norm(x, ord=2, axis=(1,2), keepdims=True)
        Nx = Gx / (tf.math.reduce_mean(Gx, axis=-1, keepdims=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(keras.Model):
    def __init__(self, dim, drop_path=0., **kwargs):
        super().__init__(**kwargs)
        self.dwconv = layers.DepthwiseConv2D(kernel_size=7, padding='same')
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.pwconv1 = layers.Dense(4 * dim)
        self.act = layers.Activation('gelu')
        self.grn = GRN(4 * dim, name=f"{kwargs['name']}_grn")
        self.pwconv2 = layers.Dense(dim)
        self.drop_path = DropPath(drop_path)

    def call(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        x = input + x
        return x 


def convnext_v2(
    input_shape=None,
    input_tensor=None, 
    num_classes=1000, 
    depths=[3,3,9,3], 
    dims=[96,192,384,768], 
    drop_path_rate=0., 
    head_init_scale=1,
    include_top=True,
    weights=None,
    model_name=None):
    
    if input_shape is None:
        input_shape = (224, 224, 3)

    if input_tensor is None:
        input_tensor = keras.Input(input_shape)

    x = input_tensor
    dp_rates = [dp for dp in np.linspace(0, drop_path_rate, sum(depths))]

    current = 0
    for i in range(4):
        if i == 0:
            x = Stem(dims[i])(x)
        else:
            x = DownSample(dims[i])(x)

        for j in range(depths[i]):
            x = Block(dims[i], dp_rates[current + j], name=f'block_{i}_{j}')(x)

        current += depths[i]

    if include_top:
        x = Head(num_classes)(x)

    outputs = x
    inputs = utils.layer_utils.get_source_inputs(input_tensor)[0]

    return keras.Model(inputs, outputs, name=model_name)


def convnextv2_atto(**kwargs):
    model = convnext_v2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], model_name='convnextv2_atto', **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = convnext_v2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], model_name='convnextv2_femto', **kwargs)
    return model

def convnextv2_pico(**kwargs):
    model = convnext_v2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], model_name='convnextv2_pico', **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = convnext_v2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], model_name='convnextv2_nano', **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = convnext_v2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], model_name='convnextv2_tiny', **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = convnext_v2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], model_name='convnextv2_base', **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = convnext_v2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], model_name='convnextv2_large', **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = convnext_v2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], model_name='convnextv2_huge', **kwargs)
    return model

