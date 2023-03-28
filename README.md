# ConvNeXt V1 & V2

The ConvNeXt family of models are a series of convolutional neural networks (ConvNets) that achieve state-of-the-art results on image classification benchmarks, and can be readily applied to plenty of other image-based tasks. For more info on these models, refer to the following papers:

- [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](http://arxiv.org/abs/2301.00808)

This repo contains unofficial implementations of both versions of ConvNeXt models in tensorflow. These are pretty much straight copies from the official PyTorch implementations ([ConvNeXt V1](https://github.com/facebookresearch/ConvNeXt) & (ConvNeXt V2)[https://github.com/facebookresearch/ConvNeXt-V2]) with minor changes to suit Keras' functional API. These implementations are capable of CPU or GPU use, and mixed-precision out of the box. Weights are currently available for ConvNeXt V1, but V2 weights are still yet to be processed. 

### Usage

For installation of tensorflow and keras, please refer to the following (guide)[https://www.tensorflow.org/install]. Once TensorFlow is installed, you can make use of the models like so:

``` py
# Contruct 'tiny' model with standard ImageNet resolution
from convnext_tf import convnext_v1

model = convnext_v1.convnext_tiny(input_shape=(224, 224, 3))
```

``` py
# Construct the model with pre-processing layers
from tensorflow import keras
from convnext_tf import convnext_v1

inputs = keras.Input((224, 224, 3))
input_tensor = keras.layers.Rescaling(1 / 255)(inputs) # Rescale inputs to 0 - 1
cxn_tiny = convnext_v1.convnext_tiny(input_tensor=input_tensor)
model = keras.Model(inputs, cxn_tiny.output)
```

``` py
# Contruct 'tiny' model with custom resolution
from convnext_tf import convnext_v1

model = convnext_v1.convnext_tiny(input_shape=(512, 512, 3))
```

``` py
# Use imagenet weights to load model
from convnext_tf import convnext_v1

model = convnext_v1.convnext_tiny(weights='imagenet')
```

``` py
# Use custom classification head
from convnext_tf import convnext_v1
from tensorflow import keras

inputs = keras.Input((224, 224, 3))
cxn_tiny = convnext_v1.convnext_tiny(input_tensor=inputs, include_top=False)
x = cxn_tiny.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dense(32, activation='relu')(x)
outputs = keras.layers.Dense(4, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)
```

### Conversion and Evaluation

Conversion scripts are `convert_weights_v1.py` and `convert_weights_v2.py` for ConvNeXt V1 and V2 models respectively. Evaluation code can be found in `eval_1k.py`. Note that the official validation ground truth labels found in the dev kit on ImageNet's website are incorrect - instead, you need to download the bounding box validation dataset from their website. The XML files in this dataset will include the correct ground truth labels which can be used for evaluation purposes. 

#### Results and pre-trained models

These are comparison results between PyTorch and TensorFlow implementations from evaluation on the ImageNet validation dataset. Converted weights for use in TensorFlow are linked in the `model` columns.

##### ImageNet-1K trained models

| name | resolution | PyTorch acc@1 | TensorFlow acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| ConvNeXt-T | 224x224 | 82.1 | 81.3 | 28M | 4.5G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth) |
| ConvNeXt-S | 224x224 | 83.1 | 82.4 | 50M | 8.7G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth) |
| ConvNeXt-B | 224x224 | 83.8 | 83.3 | 89M | 15.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth) |
| ConvNeXt-B | 384x384 | 85.1 | 84.9 | 89M | 45.0G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_384.pth) |
| ConvNeXt-L | 224x224 | 84.3 | 83.9 | 198M | 34.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth) |
| ConvNeXt-L | 384x384 | 85.5 | 85.4 | 198M | 101.0G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth) |

##### ImageNet-22K trained models

| name | resolution | PyTorch acc@1 | TensorFlow acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:| :---:|:---:|
| ConvNeXt-T | 224x224 | 82.9 | 82.3 | 29M | 4.5G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth)
| ConvNeXt-T | 384x384 | 84.1 | 84.0 | 29M | 13.1G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_384.pth)
| ConvNeXt-S | 224x224 | 84.6 | 84.1 | 50M | 8.7G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_224.pth)
| ConvNeXt-S | 384x384 | 85.8 | 85.8 | 50M | 25.5G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_384.pth)
| ConvNeXt-B | 224x224 | 85.8 | 85.4 | 89M | 15.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth)
| ConvNeXt-B | 384x384 | 86.8 | 89M | 47.0G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth)
| ConvNeXt-L | 224x224 | 86.6 | 86.4 | 198M | 34.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_224.pth)
| ConvNeXt-L | 384x384 | 87.5 | 198M | 101.0G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth)
| ConvNeXt-XL | 224x224 | 87.0 | 86.8 | 350M | 60.9G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_224_ema.pth)
| ConvNeXt-XL | 384x384 | 87.8 | 87.7 | 350M | 179.0G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth)

### Acknowledgements and References

- https://github.com/facebookresearch/ConvNeXt
- https://github.com/facebookresearch/ConvNeXt-V2
- https://github.com/rishigami/Swin-Transformer-TF
- https://github.com/sayakpaul/ConvNeXt-TF
- https://github.com/bamps53/convnext-tf