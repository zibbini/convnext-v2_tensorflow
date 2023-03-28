# Adapted from implementation by sayakpaul:
# https://github.com/sayakpaul/ConvNeXt-TF/tree/main/i1k_eval

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

from imutils import paths
import json
import re
import glob
from natsort import natsorted, ns
import xml.etree.ElementTree as ET
import convnext_tf.convnext_v1 as tf_models

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Parameters --
batch_size = 64
image_size = 384 # Set to 224 or 384
# -------------

def load_and_prepare(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)

    size = image_size
    if image_size == 224:
        crop_pct = 224 / 256
        size = int(image_size / crop_pct)        

    image = tf.image.resize(image, (size, size), method="bicubic")
    return image, label

def get_preprocessing_model():
    preprocessing_model = keras.Sequential()

    if image_size == 224:
        preprocessing_model.add(layers.CenterCrop(image_size, image_size))

    preprocessing_model.add(layers.Normalization(
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        variance=[(0.229 * 255) ** 2, (0.224 * 255) ** 2, (0.225 * 255) ** 2],
    ))

    return preprocessing_model


with open("imagenet_class_index.json", "r") as read_file:
    imagenet_labels = json.load(read_file)

MAPPING_DICT = {}
LABEL_NAMES = {}
for label_id in list(imagenet_labels.keys()):
    MAPPING_DICT[imagenet_labels[label_id][0]] = int(label_id)
    LABEL_NAMES[int(label_id)] = imagenet_labels[label_id][1]

all_val_paths = glob.glob("./ILSVRC2012_img_val/*.JPEG")
all_val_paths = natsorted(all_val_paths, alg=ns.IGNORECASE)

all_val_bbox = glob.glob("./ILSVRC2012_bbox_val_v3/val/*.xml")
all_val_bbox = natsorted(all_val_bbox, alg=ns.IGNORECASE)

all_val_labels = []
for p in all_val_bbox:
    tree = ET.parse(p)
    root = tree.getroot()
    for o in root.findall('object'):
        name = o.find('name').text
        break

    all_val_labels.append(MAPPING_DICT[name])

preprocessor = get_preprocessing_model()

dataset = tf.data.Dataset.from_tensor_slices((all_val_paths, all_val_labels))
dataset = dataset.map(load_and_prepare, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
dataset = dataset.map(lambda x, y: (preprocessor(x), y), num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

model_files = [m for m in glob.glob('./weights/*.h5') if f'{image_size}' in m]
model_files = natsorted(model_files, alg=ns.IGNORECASE)
for model in model_files:
    name = model.split('/')[-1]
    name = name.split('_')[:2]
    name = f'{name[0]}_{name[1]}'

    tf_model = tf_models.__dict__[name](input_shape=(image_size, image_size, 3))
    tf_model.load_weights(model)
    tf_model.compile(metrics=['accuracy'])

    _, accuracy = tf_model.evaluate(dataset)
    accuracy = round(accuracy * 100, 4)
    print(model, accuracy)

    del tf_model
    keras.backend.clear_session()

