import os
from typing import Iterable, Dict
from tensorflow import as tf
import kerasncp as kncp
from kerasncp.tf import LTCCell, WiredCfcCell
from tensorflow import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

DROPOUT = 0.1

DEFAULT_NCP_SEED = 22222

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
    """
    Helper function that wraps layer in a timedistributed or not depending on the arguments of this function
    """
    if not single_step:
        return keras.layers.TimeDistributed(layer)
    else:
        return layer


def generate_network_trunk(seq_len,
                           image_shape,
                           augmentation_params: Dict = None,
                           batch_size=None,
                           single_step: bool = False,
                           no_norm_layer: bool = False, ):
    """
    Generates CNN image processing backbone used in all recurrent models. Uses Keras.Functional API

    returns input to be used in Keras.Model and x, a tensor that represents the output of the network that has shape
    (batch [None], seq_len, num_units) if single step is false and (batch [None], num_units) if single step is true.
    Input has shape (batch, h, w, c) if single step is True and (batch, seq, h, w, c) otherwise

    """

    if single_step:
        inputs_image = keras.Input(shape=image_shape, name="input_image")
        inputs_value = keras.Input(shape=(2,), name="input_vector")
    else:
        inputs_image = keras.Input(batch_input_shape=(batch_size, seq_len, *image_shape), name="input_image")
        inputs_value = keras.Input(batch_input_shape=(batch_size, seq_len, 2), name="input_vector")

    xi = inputs_image
    xp = inputs_value

    if not no_norm_layer:
        xi = generate_normalization_layers(xi, single_step)

    if augmentation_params is not None:
        xi = generate_augmentation_layers(xi, augmentation_params, single_step)

    # Conv Layers
    xi = wrap_time(keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'), single_step)(
        xi)
    xi = wrap_time(keras.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'), single_step)(
        xi)
    xi = wrap_time(keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'), single_step)(
        xi)
    xi = wrap_time(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'), single_step)(
        xi)
    xi = wrap_time(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu'), single_step)(
        xi)

    xi = wrap_time(keras.layers.Flatten(), single_step)(xi)
    xi = wrap_time(keras.layers.Dense(units=128, activ+ation='linear'), single_step)(xi)
    xi = wrap_time(keras.layers.Dropout(rate=DROPOUT), single_step)(xi)

    xp = wrap_time(keras.layers.Dense(units=128, activation='relu'), single_step)(xp)
    xp = wrap_time(keras.layers.Dropout(rate=DROPOUT), single_step)(xp)

    # x = wrap_time(keras.layers.Concatenate(axis=-1), single_step)([xi, xp])
    # concatenate xi and xp using tf.concat along the last axis
    print(xi.shape, xp.shape)
    #x = wrap_time(keras.layers.Lambda(lambda y: tf.concat(y, axis=-1)), single_step)([xi, xp])
    x = tf.concat([xi, xp], axis=-1)

    return inputs_image, inputs_value, x

