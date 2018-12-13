from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from keras_finetune import restore_model_weight, create_model_info, get_np_data
from utils import *
from nets.googlenet import *
from nets.alexnet import *
from keras.utils import np_utils

def get_extract_feature_model(model, num_base_layers):
    feature_layer = model.get_layer(index=num_base_layers - 1)
    print(feature_layer.get_config())
    base_model = Model(input=model.input, output=feature_layer.output)
    print(base_model.summary())

    return base_model

def extract_features_by_batch(model, train_data, val_data, test_data):

    train_features = model.predict_on_batch(train_data)
    val_features = model.predict_on_batch(val_data)
    test_features = model.predict_on_batch(test_data)
    print ("extracted feature shape: ", train_features.shape, val_features.shape, test_features.shape)

    return train_features, val_features, test_features

def save_extracted_features(dir, architecture, pool_name, index, train_features, val_features, test_features, train_labels, val_labels, test_labels):

    data = {}
    data['train_features'] = train_features
    data['train_labels'] = train_labels

    data['val_features'] = val_features
    data['val_labels'] = val_labels

    data['test_features'] = test_features
    data['test_labels'] = test_labels

    data['index'] = str(index)
    data['architecture'] = architecture
    data['pool_name'] = pool_name

    filename = pool_name+"_"+index+"_"+architecture
    path = os.path.join (dir, filename)
    filepath = dump_pickle(data, path)
    return data, filepath

def extract_train_features_by_generator(model, pool, image_dir, model_info, is_augmented):
    train_generator = ThreadSafeGenerator(model_info, image_dir, pool['train_files'], pool['train_labels'], 1, len(pool['class_names']), False)
    print(len(pool['train_files']))
    print (len(pool['class_names']))
    train_features = []
    train_labels = []
    for i in range(len(pool['train_files'])):
        (X, Y) = next(train_generator)
        print ("predicting for X, Y of shape: ",X.shape, Y.shape)
        features = model.predict_on_batch(X)

        train_features.append(features)
        train_labels.append(Y)
    train_features = np.asarray(train_features)
    print(train_features.shape)

def main():
    model, num_base_layers, num_layers = restore_model_weight('alexnet', 10,
                                                              '/home/long/finetune/saved_models/Hela_split_30_2018-12-04_0_alexnet')
    architecture = 'alexnet'
    model_info = create_model_info(architecture)

    data_pools = load_pickle('/home/long/Desktop/Hela_split_30_2018-12-04.pickle')
    pool = data_pools['data']['0']

    is_augmented = True

    extract_model = get_extract_feature_model(model, num_base_layers)

    extract_train_features_by_generator(extract_model, pool, '/mnt/6B7855B538947C4E/Dataset/JPEG_data/Hela_JPEG', model_info, is_augmented)
    # (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = get_np_data(pool,
    #                                                                    "/mnt/6B7855B538947C4E/Dataset/JPEG_data/Hela_JPEG",
    #                                                                    model_info, is_augmented)
    # train_features, val_features, test_features = extract_features_by_batch(model, num_base_layers, X_train, X_val, X_test)
    # save_extracted_features("/home/long/Desktop", architecture, data_pools['pool_name'], '0', train_features,
    #                         val_features, test_features, Y_train, Y_val, Y_test)


if __name__ == "__main__":
    main()