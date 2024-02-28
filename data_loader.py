import os

import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib.image import imread
from tqdm import tqdm

import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_dataset_multi(root, image_size, seq_len, shift, stride, label_scale):
    file_ending = 'png'

    def sub_to_batch(sub_feature, sub_label):
        sib = sub_feature['input_image'].batch(seq_len, drop_remainder=True)
        svb = sub_feature['input_vector'].batch(seq_len, drop_remainder=True)
        slb = sub_label.batch(seq_len, drop_remainder=True)
        return tf.data.Dataset.zip(({"input_image": sib, "input_vector": svb}, slb))

    datasets = []

    output_means, output_stds = get_output_normalization(root)

    # Process single directory
    d = root  # Set the directory to the root as we are only processing one directory

    # labels = np.genfromtxt(os.path.join(d, 'data_out.csv'), delimiter=',', skip_header=1, dtype=np.float32)
    # if labels.shape[1] == 4:
    #     labels = (labels - output_means) / output_stds
    # elif labels.shape[1] == 5:
    #     labels = (labels[:, 1:] - output_means) / output_stds
    # else:
    #     raise Exception('Wrong size of input data (expected 4, got %d' % labels.shape[1])
    # labels_dataset = tf.data.Dataset.from_tensor_slices(labels)

    n_images = len([fn for fn in os.listdir(d) if fn.endswith(file_ending)])
    dataset_np = np.empty((n_images, *image_size), dtype=np.uint8)

    for ix, filename in enumerate(sorted(os.listdir(d))):
        if filename.endswith(file_ending):
            img = Image.open(os.path.join(d, filename)).convert('RGB')
            dataset_np[ix] = np.array(img)

    # dataset_vu = np.genfromtxt(os.path.join(d, 'data_in.csv'), delimiter=',', skip_header=1, dtype=np.uint8)
    # assert len(dataset_vu) == len(dataset_np), 'number of images should be equal to number of values'

    images_dataset = tf.data.Dataset.from_tensor_slices(dataset_np)
    values_dataset = tf.data.Dataset.from_tensor_slices(dataset_vu)
    dataset = tf.data.Dataset.zip(({"input_image": images_dataset, "input_vector": values_dataset}, labels_dataset))
    dataset = dataset.window(seq_len, shift=shift, stride=stride, drop_remainder=True).flat_map(sub_to_batch)
    datasets.append(dataset)

    return datasets


data_shift: int = 1
data_stride: int = 1
decay_rate: float = 0.95
val_split: float = 0.1
label_scale: float = 1
extra_data_dir: str = None

with tf.device('/cpu:0'):
    training_dataset = get_dataset_multi('Test1', IMAGE_SHAPE, 1030, data_shift, data_stride,
    val_split, label_scale,extra_data_dir)

print('\n\nTraining Dataset Size: %d\n\n' % tlen(training_dataset))

#training_dataset = training_dataset.shuffle(100).batch(batch_size)