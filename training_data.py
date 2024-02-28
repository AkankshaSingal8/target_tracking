import os
from typing import Iterable, Dict
import tensorflow as tf
import kerasncp as kncp
from kerasncp.tf import LTCCell, WiredCfcCell
from tensorflow import keras


import numpy as np
from matplotlib.image import imread
from tqdm import tqdm
from PIL import Image



IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

DROPOUT = 0.1

DEFAULT_NCP_SEED = 22222



# Shapes for generate_*_model:
# if single_step, input is tuple of image input (batch [usually 1], h, w, c), and hiddens (batch, hidden_dim)
# if not single step, is just sequence of images with shape (batch, seq_len, h, w, c) otherwise
# output is control output for not single step, for single step it is list of tensors where first element is control
# output and other outputs are any hidden states required
# if single_step, control output is (batch, 4), otherwise (batch, seq_len, 4)
# if single_step, hidden outputs typically have shape (batch, hidden_dimension)

#generate_ncp_model(1030, IMAGE_SHAPE, None, 32, DEFAULT_NCP_SEED, True, False)

def generate_ncp_model(seq_len,
                       image_shape,
                       augmentation_params=None,
                       batch_size=None,
                       seed=DEFAULT_NCP_SEED,
                       single_step: bool = False,
                       no_norm_layer: bool = False,
                       ):
    inputs_image, x = generate_network_trunk(
        seq_len,
        image_shape,
        augmentation_params=augmentation_params,
        batch_size=batch_size,
        single_step=single_step,
        no_norm_layer=no_norm_layer,
    )

    # Setup the network
    wiring = kncp.wirings.NCP(
        inter_neurons=18,  # Number of inter neurons
        command_neurons=12,  # Number of command neurons
        motor_neurons=4,  # Number of motor neurons
        sensory_fanout=6,  # How many outgoing synapses has each sensory neuron
        inter_fanout=4,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=4,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=6,  # How many incoming syanpses has each motor neuron,
        seed=seed,  # random seed to generate connections between nodes
    )

    rnn_cell = LTCCell(wiring)

    if single_step:
        inputs_state = tf.keras.Input(shape=(rnn_cell.state_size,))
        # wrap output states in list since want output to just be ndarray, not list of 1 el ndarray
        motor_out, [output_states] = rnn_cell(x, inputs_state)
        ncp_model = keras.Model([inputs_image, inputs_state], [motor_out, output_states])
    else:
        x = keras.layers.RNN(rnn_cell,
                             batch_input_shape=(batch_size,
                                                seq_len,
                                                x.shape[-1]),
                             return_sequences=True)(x)

        ncp_model = keras.Model([inputs_image], [x])

    return ncp_model




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
        inputs = keras.Input(shape=image_shape)
    else:
        inputs = keras.Input(batch_input_shape=(batch_size, seq_len, *image_shape))

    x = inputs

    if not no_norm_layer:
        x = generate_normalization_layers(x, single_step)

    if augmentation_params is not None:
        x = generate_augmentation_layers(x, augmentation_params, single_step)

    # Conv Layers
    x = wrap_time(keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'), single_step)(
        x)
    x = wrap_time(keras.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'), single_step)(
        x)
    x = wrap_time(keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'), single_step)(
        x)
    x = wrap_time(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'), single_step)(
        x)
    x = wrap_time(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu'), single_step)(
        x)

    # fully connected layers
    x = wrap_time(keras.layers.Flatten(), single_step)(x)
    x = wrap_time(keras.layers.Dense(units=128, activation='linear'), single_step)(x)
    x = wrap_time(keras.layers.Dropout(rate=DROPOUT), single_step)(x)

    return inputs, x

def get_output_normalization(root):
    training_output_mean_fn = os.path.join(root, 'stats', 'training_output_means.csv')
    if os.path.exists(training_output_mean_fn):
        print('Loading training data output means from: %s' % training_output_mean_fn)
        output_means = np.genfromtxt(training_output_mean_fn, delimiter=',')
    else:
        output_means = np.zeros(4)

    training_output_std_fn = os.path.join(root, 'stats', 'training_output_stds.csv')
    if os.path.exists(training_output_std_fn):
        print('Loading training data output std from: %s' % training_output_std_fn)
        output_stds = np.genfromtxt(training_output_std_fn, delimiter=',')
    else:
        output_stds = np.ones(4)

    return output_means, output_stds


def load_dataset_multi(root, image_size, seq_len, shift, stride, label_scale):
    file_ending = 'png'
    IMAGE_SHAPE = (144, 256, 3)
    IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

    def sub_to_batch(sub_feature, sub_label):
        sfb = sub_feature.batch(seq_len, drop_remainder=True)
        slb = sub_label.batch(seq_len, drop_remainder=True)
        return tf.data.Dataset.zip((sfb, slb))
        # return sub.batch(seq_len, drop_remainder=True)

    
    datasets = []

    #output_means, output_stds = get_output_normalization(root)

    
    labels = np.genfromtxt('Test1/data_out.csv', delimiter=',', skip_header=1, dtype=np.float32)
    print("labels", labels)
    # if labels.shape[1] == 4:
    #     labels = (labels - output_means) / output_stds
    #     # labels = labels * label_scale
    # elif labels.shape[1] == 5:
    #     labels = (labels[:, 1:] - output_means) / output_stds
    #     # labels = labels[:,1:] * label_scale
    # else:
    #     raise Exception('Wrong size of input data (expected 4, got %d' % labels.shape[1])
    
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    # n_images = len(os.listdir(os.path.join(root, d))) - 1
    n_images = len([fn for fn in os.listdir('./Test1') if file_ending in fn])
    print(n_images)
    print("no of imgs", n_images)
    # dataset_np = np.empty((n_images, 256, 256, 3), dtype=np.uint8)
    dataset_np = np.empty((n_images, *image_size), dtype=np.uint8)

    for ix in range(n_images):
        # dataset_np[ix] = imread(os.path.join(root, d, '%06d.jpeg' % ix))
        img_file_name = 'Test1/Image' + str(ix + 1) + '.'+ file_ending
        img = Image.open(img_file_name)
        img = img.resize(IMAGE_SHAPE_CV)
        # dataset_np[ix] = img[img.height - image_size[0]:, :, :]
        dataset_np[ix] = img

    hidden_state_shape = (34,) # Replace rnn_cell.state_size with actual size
    initial_hidden_states = np.zeros((n_images, *hidden_state_shape), dtype=np.float32)

    hidden_states_dataset = tf.data.Dataset.from_tensor_slices(initial_hidden_states)

    images_dataset = tf.data.Dataset.from_tensor_slices(dataset_np)
    # Modify the dataset to include hidden states
    dataset = tf.data.Dataset.zip((images_dataset, hidden_states_dataset, labels_dataset))
    dataset = dataset.map(lambda img, hidden_state, label: ((img, hidden_state), label))

    
    # dataset = tf.data.Dataset.zip((images_dataset, labels_dataset))
    # dataset = dataset.window(seq_len, shift=shift, stride=stride, drop_remainder=True).flat_map(sub_to_batch)
    # datasets.append(dataset)

    return dataset


IMAGE_SHAPE = (144, 256, 3)
shift: int = 1
stride: int = 1
decay_rate: float = 0.95
val_split: float = 0.1
label_scale: float = 1
seq_len = 1030
val_split: float = 0.1
label_scale: float = 1

training_dataset = load_dataset_multi('Test1', IMAGE_SHAPE, seq_len, shift, stride, label_scale)
#training_dataset = get_dataset_multi('Test1', IMAGE_SHAPE, seq_len, shift, stride, val_split, label_scale, extra_data_root=None)

mymodel = generate_ncp_model(1030, IMAGE_SHAPE, None, 32, DEFAULT_NCP_SEED, True, False)

decay_rate: float = 0.95
lr: float = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=500,
                                                              decay_rate=decay_rate, staircase=True)
#Adam optimizer
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
mymodel.compile(optimizer=optimizer, loss="mean_squared_error", metrics=['mse'])

epochs: int = 30
callbacks = None


training_dataset = training_dataset.batch(32)
print("Element spec of the dataset:", training_dataset.element_spec)

#setting validation data to None
history = mymodel.fit(x=training_dataset, validation_data=None, epochs=epochs,
                        use_multiprocessing=False, workers=1, max_queue_size=5, verbose=1, callbacks=callbacks)
print(history)
accuracy = mymodel.evaluate(x=training_dataset)
print('Accuracy: %.2f' % (accuracy*100))