"""Train helper functions."""
import os
import random

import numpy as np

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


def get_data_iterators(horizontal_flip=True, vertical_flip=True, width_shift_range=0.15,
                       height_shift_range=0.15, rotation_range=45, zoom_range=0.15,
                       batch_size=1, data_dir='data', target_size=(512, 512),
                       samplewise_center=False, samplewise_std_normalization=False,
                       fill_mode='constant', rescale=None, load_train_data=True,
                       color_mode='rgb'):
    """Create data iterator."""
    aug_gen = ImageDataGenerator(horizontal_flip=horizontal_flip, vertical_flip=vertical_flip,
                                 width_shift_range=width_shift_range, height_shift_range=height_shift_range,
                                 rotation_range=rotation_range, zoom_range=zoom_range,
                                 samplewise_std_normalization=samplewise_std_normalization,
                                 samplewise_center=samplewise_center, fill_mode=fill_mode,
                                 rescale=rescale)
    data_gen = ImageDataGenerator(samplewise_std_normalization=samplewise_std_normalization,
                                  samplewise_center=samplewise_center, rescale=rescale)

    if load_train_data:
        X_train, y_train = load_dataset(data_dir=os.path.join(data_dir, 'train'), target_size=target_size,
                                        color_mode=color_mode)
        train_it = aug_gen.flow(X_train, y_train, batch_size=batch_size)
    else:
        train_it = aug_gen.flow_from_directory(os.path.join(data_dir, 'train'),
                                               batch_size=batch_size, target_size=target_size,
                                               class_mode='binary', color_mode=color_mode)

    if load_train_data:
        X_val, y_val = load_dataset(data_dir=os.path.join(data_dir, 'val'), target_size=target_size,
                                    color_mode=color_mode)
        val_it = data_gen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)
    else:
        val_it = data_gen.flow_from_directory(os.path.join(data_dir, 'val'),
                                              batch_size=batch_size, target_size=target_size,
                                              class_mode='binary', color_mode=color_mode)

    test_it = data_gen.flow_from_directory(os.path.join(data_dir, 'test'),
                                           batch_size=batch_size, target_size=target_size,
                                           class_mode='binary', color_mode=color_mode,
                                           shuffle=True)

    return train_it, val_it, test_it


def load_dataset(data_dir='data', color_mode='rgb', target_size=(512, 512)):
    """Load dataset into memory."""
    imgs = []
    y = []

    classes = os.listdir(data_dir)
    classes.sort()
    for i, c in enumerate(classes):
        c_imgs = os.listdir(os.path.join(data_dir, c))
        imgs.extend([os.path.join(data_dir, c, c_img) for c_img in c_imgs])
        y.extend([i]*len(c_imgs))

    N = len(imgs)
    idx = range(N)
    random.shuffle(idx)

    imgs = np.array(imgs)
    y = np.array(y, dtype=np.int8)

    imgs = imgs[idx]
    y = y[idx]

    grayscale = False
    channels = 3
    if color_mode == 'grayscale':
        grayscale = True
        channels = 1

    X = np.zeros((N, channels) + target_size, dtype=np.float32)
    for i, img_path in enumerate(imgs):
        img = load_img(img_path, grayscale=grayscale, target_size=target_size)
        X[i] = img_to_array(img)

    return X, y
