def load_dataset_multi(root, image_size, seq_len, shift, stride, label_scale):
    file_ending = 'png'

    def sub_to_batch(sub_feature, sub_label):
        sfb = sub_feature.batch(seq_len, drop_remainder=True)
        slb = sub_label.batch(seq_len, drop_remainder=True)
        return tf.data.Dataset.zip((sfb, slb))
        # return sub.batch(seq_len, drop_remainder=True)

    dirs = sorted(os.listdir(root))
    dirs = [d for d in dirs if 'cached' not in d and 'stats' not in d]
    datasets = []

    output_means, output_stds = get_output_normalization(root)

    for (run_number, d) in tqdm(enumerate(dirs)):
        labels = np.genfromtxt(os.path.join(root, d, 'data_out.csv'), delimiter=',', skip_header=1, dtype=np.float32)

        if labels.shape[1] == 4:
            labels = (labels - output_means) / output_stds
            # labels = labels * label_scale
        elif labels.shape[1] == 5:
            labels = (labels[:, 1:] - output_means) / output_stds
            # labels = labels[:,1:] * label_scale
        else:
            raise Exception('Wrong size of input data (expected 4, got %d' % labels.shape[1])
        labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
        # n_images = len(os.listdir(os.path.join(root, d))) - 1
        n_images = len([fn for fn in os.listdir(os.path.join(root, d)) if file_ending in fn])
        # dataset_np = np.empty((n_images, 256, 256, 3), dtype=np.uint8)
        dataset_np = np.empty((n_images, *image_size), dtype=np.uint8)

        for ix in range(n_images):
            # dataset_np[ix] = imread(os.path.join(root, d, '%06d.jpeg' % ix))
            img = Image.open(os.path.join(root, d, '%06d.%s' % (ix, file_ending)))
            # dataset_np[ix] = img[img.height - image_size[0]:, :, :]
            dataset_np[ix] = img

        images_dataset = tf.data.Dataset.from_tensor_slices(dataset_np)
        dataset = tf.data.Dataset.zip((images_dataset, labels_dataset))
        dataset = dataset.window(seq_len, shift=shift, stride=stride, drop_remainder=True).flat_map(sub_to_batch)
        datasets.append(dataset)

    return datasets


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
    
    training_datasets = ds[val_ix:]

    # if either dataset has length 0, trying to call flat map raises error that return type is wrong
    assert len(training_datasets) > 0 and len(validation_datasets) > 0, f"Training or validation dataset has no points!" \
                                                                        f"Train dataset len: {len(training_datasets)}" \
                                                                        f"Val dataset len: {len(validation_datasets)}"
                                                                        
    training = tf.data.Dataset.from_tensor_slices(training_datasets).flat_map(lambda x: x)
    validation = tf.data.Dataset.from_tensor_slices(validation_datasets).flat_map(lambda x: x)

    return training, validation