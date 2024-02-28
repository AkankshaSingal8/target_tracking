import os

import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib.image import imread
from tqdm import tqdm


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

    images_dataset = tf.data.Dataset.from_tensor_slices(dataset_np)
    dataset = tf.data.Dataset.zip((images_dataset, labels_dataset))
    #dataset = dataset.window(seq_len, shift=shift, stride=stride, drop_remainder=True).flat_map(sub_to_batch)
    #datasets.append(dataset)

    return dataset

def get_dataset_multi(root, image_size, seq_len, shift, stride, validation_ratio, label_scale, extra_data_root=None):
    ds = load_dataset_multi(root, image_size, seq_len, shift, stride, label_scale)
    print('n bags: %d' % len(ds))
    cnt = 0

    for d in ds:
        for (ix, _) in enumerate(d):
            pass
            cnt += ix
    print('n windows: %d' % cnt)

    

    val_ix = int(len(ds) * validation_ratio)
    print('\nval_ix: %d\n' % val_ix)
    #validation_datasets = ds[:val_ix]

    training_datasets = ds[val_ix:]

    # if either dataset has length 0, trying to call flat map raises error that return type is wrong
    # assert len(training_datasets) > 0 and len(validation_datasets) > 0, f"Training or validation dataset has no points!" \
    #                                                                     f"Train dataset len: {len(training_datasets)}" \
    #                                                                     f"Val dataset len: {len(validation_datasets)}"
    training = tf.data.Dataset.from_tensor_slices(training_datasets).flat_map(lambda x: x)
    #validation = tf.data.Dataset.from_tensor_slices(validation_datasets).flat_map(lambda x: x)

    #return training, validation
    return training

IMAGE_SHAPE = (144, 256, 3)
shift: int = 1
stride: int = 1
decay_rate: float = 0.95
val_split: float = 0.1
label_scale: float = 1
seq_len = 1030
val_split: float = 0.1
label_scale: float = 1

datasets = load_dataset_multi('Test1', IMAGE_SHAPE, seq_len, shift, stride, label_scale)
#training_dataset = get_dataset_multi('Test1', IMAGE_SHAPE, seq_len, shift, stride, val_split, label_scale, extra_data_root=None)
print('load dataset shape', datasets.element_spec)