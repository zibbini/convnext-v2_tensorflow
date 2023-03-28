# Adapted from implementations by sayakpaul and bamps53:
# https://github.com/sayakpaul/ConvNeXt-TF/blob/main/convert.py
# https://github.com/bamps53/convnext-tf/blob/master/convert_weights.py

import os
import cv2
import numpy as np
import timm
import torch
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

import convnext_pt.convnext as pt_models
import convnext_tf.convnext_v1 as tf_models

model_urls = [
    # ImageNet-1k trained ------------------------------------------------
    'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_384.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth',
    # ImageNet-22k trained -----------------------------------------------
    'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_384.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_224.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_384.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_224.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_224_ema.pth',
    'https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth',
]


def load_test_image(image_size=(224, 224)):
    from skimage.data import chelsea
    img = chelsea()  # Chelsea the cat
    img = cv2.resize(img, image_size)
    img = img / 255
    img = (img - np.array([0.485, 0.456, 0.406])) / \
        np.array([0.229, 0.224, 0.225])
    return img


def load_from_url(model, url):
    checkpoint = torch.hub.load_state_dict_from_url(
        url=url, map_location='cpu', check_hash=True)
    model.load_state_dict(checkpoint['model'])
    return model


def test_models(pt_model, tf_model, img_size):
    img = load_test_image(img_size)
    pt_x = torch.tensor(img[None, ]).permute(0, 3, 1, 2).to(torch.float32)
    with torch.no_grad():
        pt_y = torch.softmax(pt_model(pt_x), -1).numpy()

    tf_x = tf.convert_to_tensor(img[None, ])
    tf_y = tf.nn.softmax(tf_model(tf_x)).numpy()
    try:
        np.testing.assert_allclose(pt_y, tf_y, rtol=1e-3)
    except Exception as e:
        print(e)


def main():
    os.makedirs('weights', exist_ok=True)
    for model_url in model_urls:
        file_name = os.path.basename(model_url)
        model_name = '_'.join(file_name.split('_')[:2])

        input_shape = (384, 384, 3) if '384' in file_name else (224, 224, 3)

        save_name = file_name.replace('pth', 'h5')
        if os.path.exists(f'weights/{save_name}'):
            continue

        pt_model = pt_models.__dict__[model_name]()
        pt_model = load_from_url(pt_model, model_url)
        pt_model.eval()

        tf_model = tf_models.__dict__[model_name](input_shape=input_shape)

        assert tf_model.count_params() == sum(
            p.numel() for p in pt_model.parameters()
        )

        param_list = list(pt_model.parameters())
        model_states = pt_model.state_dict()
        state_list = list(model_states.keys())

        stem_block = tf_model.get_layer('stem')

        for layer in stem_block.layers:
            if isinstance(layer, layers.Conv2D):
                layer.kernel.assign(
                    tf.Variable(model_states['downsample_layers.0.0.weight'].detach().numpy().transpose(2, 3, 1, 0))
                )
                layer.bias.assign(tf.Variable(model_states['downsample_layers.0.0.bias'].detach().numpy()))
            elif isinstance(layer, layers.LayerNormalization):
                layer.gamma.assign(tf.Variable(model_states['downsample_layers.0.1.weight'].detach().numpy()))
                layer.beta.assign(tf.Variable(model_states['downsample_layers.0.1.bias'].detach().numpy()))

        # Downsampling layers.
        for i in range(3):
            downsampling_block = tf_model.get_layer('down_sample' if i == 0 else f'down_sample_{i}')
            pytorch_layer_prefix = f'downsample_layers.{i + 1}'

            for l in downsampling_block.layers:
                if isinstance(l, layers.LayerNormalization):
                    l.gamma.assign(
                        tf.Variable(
                            model_states[f'{pytorch_layer_prefix}.0.weight'].detach().numpy()
                        )
                    )
                    l.beta.assign(
                        tf.Variable(model_states[f'{pytorch_layer_prefix}.0.bias'].detach().numpy())
                    )
                elif isinstance(l, layers.Conv2D):
                    l.kernel.assign(
                        tf.Variable(
                            model_states[f'{pytorch_layer_prefix}.1.weight']
                            .detach().numpy()
                            .transpose(2, 3, 1, 0)
                        )
                    )
                    l.bias.assign(
                        tf.Variable(model_states[f'{pytorch_layer_prefix}.1.bias'].detach().numpy())
                    )


        for m in range(4):
            num_blocks = len([layer for layer in tf_model.layers if f'block_{m}' in layer.name])

            for i in range(num_blocks):
                stage_block = tf_model.get_layer(f'block_{m}_{i}')
                stage_prefix = f'stages.{m}.{i}'

                for j, layer in enumerate(stage_block.layers):
                    if isinstance(layer, layers.DepthwiseConv2D):
                        layer.depthwise_kernel.assign(
                            tf.Variable(
                                model_states[f'{stage_prefix}.dwconv.weight']
                                .detach().numpy()
                                .transpose(2, 3, 0, 1)
                            )
                        )
                        layer.bias.assign(
                            tf.Variable(model_states[f'{stage_prefix}.dwconv.bias'].detach().numpy())
                        )
                    elif isinstance(layer, layers.Dense):
                        if j == 2:
                            layer.kernel.assign(
                                tf.Variable(
                                    model_states[f'{stage_prefix}.pwconv1.weight']
                                    .detach().numpy()
                                    .transpose([1, 0])
                                )
                            )
                            layer.bias.assign(
                                tf.Variable(
                                    model_states[f'{stage_prefix}.pwconv1.bias'].detach().numpy()
                                )
                            )
                        elif j == 4:
                            layer.kernel.assign(
                                tf.Variable(
                                    model_states[f'{stage_prefix}.pwconv2.weight']
                                    .detach().numpy()
                                    .transpose([1, 0])
                                )
                            )
                            layer.bias.assign(
                                tf.Variable(
                                    model_states[f'{stage_prefix}.pwconv2.bias'].detach().numpy()
                                )
                            )
                    elif isinstance(layer, layers.LayerNormalization):
                        layer.gamma.assign(
                            tf.Variable(model_states[f'{stage_prefix}.norm.weight'].detach().numpy())
                        )
                        layer.beta.assign(
                            tf.Variable(model_states[f'{stage_prefix}.norm.bias'].detach().numpy())
                        )
                    elif isinstance(layer, tf_models.LayerScale):
                        layer.gamma.assign(
                            tf.Variable(model_states[f'{stage_prefix}.gamma'].detach().numpy())
                        )

        head_block = tf_model.get_layer('head')
        for layer in head_block.layers:
            if isinstance(layer, layers.LayerNormalization):
                layer.gamma.assign(tf.Variable(model_states['norm.weight'].detach().numpy()))
                layer.beta.assign(tf.Variable(model_states['norm.bias'].detach().numpy()))
            elif isinstance(layer, layers.Dense):
                layer.kernel.assign(
                    tf.Variable(model_states['head.weight'].detach().numpy().transpose([1, 0]))
                )
                layer.bias.assign(
                    tf.Variable(model_states['head.bias'].detach().numpy())
                )

        print(f'successfully converted {model_name}!')
        test_models(pt_model, tf_model, input_shape[:-1])

        tf_model.save_weights(f'weights/{save_name}')
        del tf_model
        keras.backend.clear_session()

if __name__ == '__main__':
    main()