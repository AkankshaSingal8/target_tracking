import os
from typing import Iterable, Dict
import kerasncp as kncp
from kerasncp.tf import LTCCell, WiredCfcCell
from tensorflow import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

DROPOUT = 0.1
DEFAULT_CFC_CONFIG = {
    "clipnorm": 1,
    "backbone_activation": "silu",
    "backbone_dr": 0.1,
    "forget_bias": 1.6,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 1e-06
}
DEFAULT_NCP_SEED = 22222
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Shapes for generate_*_model:
# if single_step, input is tuple of image input (batch [usually 1], h, w, c), and hiddens (batch, hidden_dim)
# if not single step, is just sequence of images with shape (batch, seq_len, h, w, c) otherwise
# output is control output for not single step, for single step it is list of tensors where the first element is control
# output and other outputs are any hidden states required
# if single_step, control output is (batch, 4), otherwise (batch, seq_len, 4)
# if single_step, hidden outputs typically have shape (batch, hidden_dimension)

def generate_augmentation_layers(x, augmentation_params: Dict, single_step: bool):
    # translate -> rotate -> zoom -> noise
    trans = augmentation_params.get('translation', None)
    rot = augmentation_params.get('rotation', None)
    zoom = augmentation_params.get('zoom', None)
    noise = augmentation_params.get('noise', None)

    if trans is not None:
        x = wrap_time(keras.layers.experimental.preprocessing.RandomTranslation(
            height_factor=trans, width_factor=trans), single_step)(x)

    if rot is not None:
        x = wrap_time(keras.layers.experimental.preprocessing.RandomRotation(rot), single_step)(x)

    if zoom is not None:
        x = wrap_time(keras.layers.experimental.preprocessing.RandomZoom(
            height_factor=zoom, width_factor=zoom), single_step)(x)

    if noise:
        x = wrap_time(keras.layers.GaussianNoise(stddev=noise), single_step)(x)

    return x

def generate_normalization_layers(x, single_step: bool):
    rescaling_layer = keras.layers.experimental.preprocessing.Rescaling(1. / 255)

    normalization_layer = keras.layers.experimental.preprocessing.Normalization(
        mean=[0.41718618, 0.48529191, 0.38133072],
        variance=[.057, .05, .061])

    x = rescaling_layer(x)
    x = wrap_time(normalization_layer, single_step)(x)
    return x

def wrap_time(layer, single_step: bool):
    if not single_step:
        return keras.layers.TimeDistributed(layer)
    else:
        return layer

def generate_network_trunk(seq_len,
                           image_shape,
                           augmentation_params: Dict = None,
                           batch_size=None,
                           single_step: bool = False,
                           no_norm_layer: bool = False):
    if single_step:
        inputs = keras.Input(shape=image_shape)
    else:
        inputs = keras.Input(batch_input_shape=(batch_size, seq_len, *image_shape))

    x = inputs

    if not no_norm_layer:
        x = generate_normalization_layers(x, single_step)

    if augmentation_params is not None:
        x = generate_augmentation_layers(x, augmentation_params, single_step)

    # Conv Layers
    x = wrap_time(keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'), single_step)(x)
    x = wrap_time(keras.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'), single_step)(x)
    x = wrap_time(keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'), single_step)(x)
    x = wrap_time(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'), single_step)(x)
    x = wrap_time(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu'), single_step)(x)

    return inputs, x

image_path = 'Image01.png'
image = Image.open(image_path)
image = image.resize(IMAGE_SHAPE_CV)
print("Image size after resize:", image.size)
image_array = np.array(image)
print("image array shape", image_array.shape)
image_array = np.expand_dims(image_array, axis=0)  # Add a batch dimension
print("image array shape after adding dim", image_array.shape)

# Define the image shape and other parameters
image_shape = IMAGE_SHAPE
seq_len = 1

# Generate the input and output using generate_network_trunk
inputs, trunk_output = generate_network_trunk(seq_len, image_shape, augmentation_params=None, batch_size=32, single_step=True)

# Create a model using the inputs and trunk_output
model = keras.Model(inputs=inputs, outputs=trunk_output)

# Now you can use this model for your specific task, for example, image processing.
output = model.predict(image_array)

# Print the shape of the output
print("Output shape:", output.shape)

# Visualize all 16 feature maps
num_rows, num_cols = 4, 4  # Assuming 16 channels
feature_maps = output[0, :, :, :]

# Create a figure and a set of subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

# Flatten the axes array to iterate over the subplots
axes = axes.flatten()

for i in range(feature_maps.shape[-1]):
    ax = axes[i]
    ax.imshow(feature_maps[:, :, i], cmap='viridis')
    ax.axis('off')
    ax.set_title(f'Channel {i}')

plt.tight_layout()
plt.show()
