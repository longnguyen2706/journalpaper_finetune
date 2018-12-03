# -*- coding: utf-8 -*-
import numpy as np
from utils.customlayers import crosschannelnormalization
from utils.customlayers import Softmax4D
from utils.customlayers import splittensor
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import merge
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import SGD
from scipy.misc import imread
from scipy.misc import imresize
import theano.ifelse

def convnet(network, weights_path=None, heatmap=False, trainable=None):
    """
    Returns a keras model for a CNN.

    BEWARE !! : Since the different convnets have been trained in different settings, they don't take
    data of the same shape. You should change the arguments of preprocess_image_batch for each CNN :
    * For AlexNet, the data are of shape (227,227), and the colors in the RGB order (default)
    * For VGG16 and VGG19, the data are of shape (224,224), and the colors in the BGR order

    It can also be used to look at the hidden layers of the model.

    It can be used that way :
    >>> im = preprocess_image_batch(['cat.jpg'])

    >>> # Test pretrained model
    >>> model = convnet('vgg_16', 'weights/vgg16_weights.h5')
    >>> sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    >>> model.compile(optimizer=sgd, loss='categorical_crossentropy')
    >>> out = model.predict(im)

    Parameters
    --------------
    network: str
        The type of network chosen. For the moment, can be 'vgg_16' or 'vgg_19'

    weights_path: str
        Location of the pre-trained model. If not given, the model will be trained

    heatmap: bool
        Says wether the fully connected layers are transformed into Convolution2D layers,
        to produce a heatmap instead of a


    Returns
    ---------------
    model:
        The keras model for this convnet

    output_dict:
        Dict of feature layers, asked for in output_layers.
    """
    def __get_heatmap_model():
        convnet_heatmap = convnet_init(heatmap=True)
        for layer in convnet_heatmap.layers:
            if layer.name.startswith('conv'):
                orig_layer = convnet.get_layer(layer.name)
                layer.set_weights(orig_layer.get_weights())
            elif layer.name.startswith('dense'):
                orig_layer = convnet.get_layer(layer.name)
                W, b = orig_layer.get_weights()
                n_filter, previous_filter, ax1, ax2 = layer.get_weights()[0].shape
                new_W = W.reshape((previous_filter, ax1, ax2, n_filter))
                new_W = new_W.transpose((3, 0, 1, 2))
                new_W = new_W[:, :, ::-1, ::-1]
                layer.set_weights([new_W, b])
        return convnet_heatmap

    # Select the network
    convnet_init = __get_model_based_on_input_network(network)
    convnet = convnet_init(weights_path, heatmap=False)
    return __get_heatmap_model() if heatmap else convnet


def __get_model_based_on_input_network(network):
    """
    Select correct model method based on input string

    :type network: str
    """
    if network == 'alexnet':
        convnet_init = AlexNet
    else:
        raise ValueError("Only 'alexnet' models available")
    return convnet_init

def AlexNet(weights_path=None, heatmap=False):
    if heatmap:
        inputs = Input(shape=(3, None, None))
    else:
        inputs = Input(shape=(3, 227, 227))

    conv_1 = Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = crosschannelnormalization(name='convpool_1')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = merge([
                       Convolution2D(128, 5, 5, activation='relu', name='conv_2_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_2)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_2')

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Convolution2D(384, 3, 3, activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = merge([
                       Convolution2D(192, 3, 3, activation='relu', name='conv_4_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_4)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_4')

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = merge([
                       Convolution2D(128, 3, 3, activation='relu', name='conv_5_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_5)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_5')

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)

    if heatmap:
        dense_1 = Convolution2D(4096, 6, 6, activation='relu', name='dense_1')(dense_1)
        dense_2 = Convolution2D(4096, 1, 1, activation='relu', name='dense_2')(dense_1)
        dense_3 = Convolution2D(1000, 1, 1, name='dense_3')(dense_2)
        prediction = Softmax4D(axis=1, name='softmax')(dense_3)
    else:
        dense_1 = Flatten(name='flatten')(dense_1)
        dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        dense_3 = Dense(1000, name='dense_3')(dense_3)
        prediction = Activation('softmax', name='softmax')(dense_3)

    model = Model(input=inputs, output=prediction)

    if weights_path:
        model.load_weights(weights_path)
    return model


if __name__ == '__main__':
    alexnet_weight = "/mnt/6B7855B538947C4E/deeplearning/pretrained_weights/alexnet_weights.h5"
    alexnet = convnet('alexnet', alexnet_weight, False, None)
    print(alexnet.summary())
