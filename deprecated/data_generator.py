
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.preprocessing.sequence
from keras_preprocessing import get_keras_submodule

import numpy as np
from utils import *


def abstractmethod(funcobj):
    """A decorator indicating abstract methods.

    Requires that the metaclass is ABCMeta or derived from it.  A
    class that has a metaclass derived from ABCMeta cannot be
    instantiated unless all of its abstract methods are overridden.
    The abstract methods can be called using any of the normal
    'super' call mechanisms.

    Usage:

        class C(metaclass=ABCMeta):
            @abstractmethod
            def my_abstract_method(self, ...):
                ...
    """
    funcobj.__isabstractmethod__ = True
    return funcobj

class Sequence(object):
    """Base object for fitting to a sequence of data, such as a dataset.

    Every `Sequence` must implements the `__getitem__` and the `__len__` methods.
    If you want to modify your dataset between epochs you may implement `on_epoch_end`.
    The method `__getitem__` should return a complete batch.

    # Notes

    `Sequence` are a safer way to do multiprocessing. This structure guarantees that the network will only train_by_fit_generator once
     on each sample per epoch which is not the case with generators.

    # Examples

    ```python
        from skimage.io import imread
        from skimage.transform import resize
        import numpy as np

        # Here, `x_set` is list of path to the images
        # and `y_set` are the associated classes.

        class CIFAR10Sequence(Sequence):

            def __init__(self, x_set, y_set, batch_size):
                self.x, self.y = x_set, y_set
                self.batch_size = batch_size

            def __len__(self):
                return np.ceil(len(self.x) / float(self.batch_size))

            def __getitem__(self, idx):
                batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

                return np.array([
                    resize(imread(file_name), (200, 200))
                       for file_name in batch_x]), np.array(batch_y)
    ```
    """

    @abstractmethod
    def __getitem__(self, index):
        """Gets batch at position `index`.

        # Arguments
            index: position of the batch in the Sequence.

        # Returns
            A batch
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.
        """
        raise NotImplementedError

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pass

    def __iter__(self):
        """Create an infinite generator that iterate over the Sequence."""
        while True:
            for item in (self[i] for i in range(len(self))):
                yield item

#TODO: fix the bug in shuffle=True
# see https://github.com/keras-team/keras/issues/9707
class DataGenerator(Sequence):
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
        return prepare_image_data_arr_and_label(self.image_dir, batch_x, 227, 227, 0, 0, batch_y)

    def generator(self):
        pass

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


if __name__ == '__main__':
    data_pools = load_pickle('/home/long/Desktop/Hela_split_30_2018-12-04.pickle')
    pool = data_pools['data']['0']
    print(pool['data_name'])
    print(len(pool['train_files']))
    num_classes = len(pool['class_names'])
    train_generator, val_generator, test_generator = get_generators(pool,  "/mnt/6B7855B538947C4E/Dataset/JPEG_data/Hela_JPEG", 16, 32)
    next(train_generator)



def data_gen(top_dim, bot_dim):
    """
    Generator to yield batches of two inputs (per sample) with shapes top_dim and
    bot_dim along with their labels.
    """
    batch_size = 264
    while True:
        top_batch = []
        bot_batch = []
        batch_labels = []
        for i in range(batch_size):
            # Create random arrays
            rand_pix = np.random.randint(100, 256)
            top_img = np.full(top_dim, rand_pix)
            bot_img = np.full(bot_dim, rand_pix)

            # Set a label
            label = np.random.choice([0, 1])
            batch_labels.append(label)

            # Pack each input image separately
            top_batch.append(top_img)
            bot_batch.append(bot_img)

        yield [np.array(top_batch), np.array(bot_batch)], np.array(batch_labels)
