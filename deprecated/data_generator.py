
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_preprocessing import get_keras_submodule

import numpy as np
from utils import *

kerasUtils = get_keras_submodule('utils')


#TODO: fix the bug in shuffle=True
# see https://github.com/keras-team/keras/issues/9707
class DataGenerator(kerasUtils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_images, labels, num_classes, image_dir, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.labels = labels
        self.list_images = list_images
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.image_dir = image_dir
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_images ) / self.batch_size))

    def __getitem__(self, idx):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_x = self.list_images[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Generate data
        # print(indexes)
        batch_data, batch_labels = self.__data_generation(batch_x, batch_y)
        return batch_data, batch_labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # self.list_images = np.arange(len(self.list_images))
        if self.shuffle == True:
            np.random.shuffle(self.list_images)

    def __data_generation(self, batch_x, batch_y):
        batch_data =  prepare_numpy_data_arr(self.image_dir, batch_x)
        return batch_data, kerasUtils.toCategorial(batch_y, num_classes=self.num_classes)


def get_generators(split, image_dir, train_batch, test_batch):
    train_images = split['train_files']
    train_labels = split['train_labels']

    val_images = split['val_files']
    val_labels = split['val_labels']

    test_images = split['test_files']
    test_labels = split['test_labels']
    num_classes = len(split['class_names'])


    train_generator = DataGenerator(train_images, train_labels, num_classes, image_dir, train_batch)
    val_generator = DataGenerator(test_images, test_labels, num_classes, image_dir, test_batch)
    test_generator = DataGenerator(val_images, val_labels, num_classes, image_dir, test_batch)

    return train_generator, val_generator, test_generator
