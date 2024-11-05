import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras
from keras import utils


class DropPath(layers.Layer):
    # borrowed from https://github.com/rishigami/Swin-Transformer-TF/blob/main/swintransformer/model.py
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    @staticmethod
    def drop_path(x, drop_prob, is_training):
        if (not is_training) or (drop_prob == 0.):
            return x

        # Compute keep_prob
        keep_prob = 1.0 - drop_prob

        # Compute drop_connect tensor
        random_tensor = keep_prob
        shape = (tf.shape(x)[0],) + (1,) * \
            (len(tf.shape(x)) - 1)
        random_tensor += tf.random.uniform(shape, dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)
        output = tf.math.divide(x, keep_prob) * binary_tensor
        return output

    def call(self, x, training=None):
        return self.drop_path(x, self.drop_prob, training)


class LayerScale(layers.Layer):
    def __init__(self, dim, init_value=1e-6, name='layer_scale', **kwargs):
        super().__init__(**kwargs)
        self.gamma = tf.Variable(
            initial_value=init_value * tf.ones((dim), dtype=self.compute_dtype),
            trainable=True,
            name=f'{name}/gamma',
            dtype=self.compute_dtype) if init_value > 0 else None

    def call(self, x):
        if self.gamma is not None:
            x = self.gamma * x
        return x   


class DownSample(keras.Model):
    def __init__(self, dim):
        super().__init__()
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.conv = layers.Conv2D(dim, kernel_size=2, strides=2, padding='same')

    def call(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x


class Stem(keras.Model):
    def __init__(self, dim):
        super().__init__()
        self.conv = layers.Conv2D(dim, kernel_size=4, strides=4, padding='valid')
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class Block(keras.Model):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.dwconv = layers.DepthwiseConv2D(kernel_size=7, padding='same')
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.pwconv1 = layers.Dense(4 * dim)
        self.act = layers.Activation('gelu')
        self.pwconv2 = layers.Dense(dim)
        self.layer_scale = LayerScale(dim, layer_scale_init_value, name=f"{kwargs['name']}_layer_scale")
        self.drop_path = DropPath(drop_path)

    def call(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.layer_scale(x)
        x = self.drop_path(x)
        x = input + x
        return x       


class Head(keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.norm = layers.LayerNormalization()
        self.predictions = layers.Dense(num_classes)

    def call(self, x):
        x = self.avg_pool(x)
        x = self.norm(x)
        x = self.predictions(x)
        return x


weights_1k = {
    'convnext_tiny_224': 'https://github.com/zibbini/convnext-v2_tensorflow/releases/download/v0.1/convnext_tiny_1k_224_ema.h5',
    'convnext_small_224': 'https://github.com/zibbini/convnext-v2_tensorflow/releases/download/v0.1/convnext_small_1k_224_ema.h5',
    'convnext_base_224': 'https://github.com/zibbini/convnext-v2_tensorflow/releases/download/v0.1/convnext_base_1k_224_ema.h5',
    'convnext_base_384': 'https://github.com/zibbini/convnext-v2_tensorflow/releases/download/v0.1/convnext_base_1k_384_ema.h5',
    'convnext_large_224': 'https://github.com/zibbini/convnext-v2_tensorflow/releases/download/v0.1/convnext_large_1k_224_ema.h5',
    'convnext_large_384': 'https://github.com/zibbini/convnext-v2_tensorflow/releases/download/v0.1/convnext_large_1k_384_ema.h5'
}

weights_22k = {
    'convnext_tiny_224': 'https://github.com/zibbini/convnext-v2_tensorflow/releases/download/v0.1/convnext_tiny_22k_1k_224.h5',
    'convnext_tiny_384': 'https://github.com/zibbini/convnext-v2_tensorflow/releases/download/v0.1/convnext_tiny_22k_1k_384.h5',
    'convnext_small_224': 'https://github.com/zibbini/convnext-v2_tensorflow/releases/download/v0.1/convnext_small_22k_1k_224_ema.h5',
    'convnext_small_384': 'https://github.com/zibbini/convnext-v2_tensorflow/releases/download/v0.1/convnext_small_22k_1k_384_ema.h5',
    'convnext_base_224': 'https://github.com/zibbini/convnext-v2_tensorflow/releases/download/v0.1/convnext_base_22k_1k_224_ema.h5',
    'convnext_base_384': 'https://github.com/zibbini/convnext-v2_tensorflow/releases/download/v0.1/convnext_base_22k_1k_384_ema.h5',
    'convnext_large_224': 'https://github.com/zibbini/convnext-v2_tensorflow/releases/download/v0.1/convnext_large_22k_1k_224_ema.h5',
    'convnext_large_384': 'https://github.com/zibbini/convnext-v2_tensorflow/releases/download/v0.1/convnext_large_22k_1k_384_ema.h5',    
    'convnext_xlarge_224': 'https://github.com/zibbini/convnext-v2_tensorflow/releases/download/v0.1/convnext_xlarge_22k_1k_224_ema.h5',   
    'convnext_xlarge_384': 'https://github.com/zibbini/convnext-v2_tensorflow/releases/download/v0.1/convnext_xlarge_22k_1k_384_ema.h5'
}


def convnext(
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

    model = keras.Model(inputs, outputs, name=model_name)
    if weights is not None:
        key = f'{model_name}_{input_shape[0]}'
        url = weights_1k[key] if weights == 'imagenet_1k' else weights_22k[key]
        pretrained_weights = keras.utils.get_file(origin=url)
        model.load_weights(pretrained_weights)

    return model


def convnext_atto(**kwargs):
    model = convnext(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], model_name='convnext_atto', **kwargs)
    return model

def convnext_femto(**kwargs):
    model = convnext(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], model_name='convnext_femto', **kwargs)
    return model

def convnext_pico(**kwargs):
    model = convnext(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], model_name='convnext_pico', **kwargs)
    return model

def convnext_nano(**kwargs):
    model = convnext(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], model_name='convnext_nano', **kwargs)
    return model

def convnext_tiny(**kwargs):
    model = convnext(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], model_name='convnext_tiny', **kwargs)
    return model

def convnext_small(**kwargs):
    model = convnext(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], model_name='convnext_small', **kwargs)
    return model

def convnext_base(**kwargs):
    model = convnext(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], model_name='convnext_base', **kwargs)
    return model

def convnext_large(**kwargs):
    model = convnext(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], model_name='convnext_large', **kwargs)
    return model

def convnext_xlarge(**kwargs):
    model = convnext(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], model_name='convnext_xlarge', **kwargs)
    return model
