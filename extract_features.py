from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from keras_finetune import restore_model_weight, create_model_info
from utils import get_np_data
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
    print("extracted feature shape: ", train_features.shape, val_features.shape, test_features.shape)

    return train_features, val_features, test_features


def save_features_and_prediction(save_dir, architecture, model_path, image_dir, pools, pool_idx, is_augmented):
    data = {}
    pool_idx = str(pool_idx)
    pool = pools['data'][pool_idx]
    num_classes = len(pool['class_names'])

    model, num_base_layers, num_layers = restore_model_weight(architecture, num_classes, model_path)

    model_info = create_model_info(architecture)

    extract_model = get_extract_feature_model(model, num_base_layers)

    train_prediction, val_prediction, test_prediction, train_labels, val_labels, test_labels = get_pool_prediction(
        model, image_dir, pool, model_info, is_augmented)

    train_features, val_features, test_features, _, _, _ = get_pool_prediction(extract_model, image_dir, pool,
                                                                               model_info, is_augmented)

    data['train_features'] = train_features
    data['train_labels'] = train_labels
    data['train_prediction'] = train_prediction

    data['val_features'] = val_features
    data['val_labels'] = val_labels
    data['val_prediction'] = val_prediction

    data['test_features'] = test_features
    data['test_labels'] = test_labels
    data['test_prediction'] = test_prediction

    data['class_names'] = pool['class_names']
    data['index'] = pool_idx
    data['architecture'] = architecture
    data['pool_name'] = pool['data_name']

    filename = data['pool_name'] + "_" + pool_idx + "_" + architecture
    path = os.path.join(save_dir, filename)
    filepath = dump_pickle(data, path)
    return data, filepath


def get_prediction_by_generator(model, image_dir, short_paths, labels, num_class, model_info, is_augmented):
    generator = ThreadSafeGenerator(model_info, image_dir, short_paths, labels, 1, num_class, is_augmented)
    feature_arr = []
    label_arr = []
    for i in range(len(short_paths)):
        (X, Y) = next(generator)
        print("predicting for X, Y of shape: ", X.shape, Y.shape)
        feature = model.predict_on_batch(X)

        feature_arr.append(feature)
        label_arr.append(Y)
    feature_arr = np.asarray(feature_arr)
    label_arr = np.asarray(label_arr)
    print(feature_arr.shape, label_arr.shape)
    return feature_arr, label_arr


def get_pool_prediction(model, image_dir, pool, model_info, is_augmented):
    train_features, train_labels = get_prediction_by_generator(model, image_dir, pool['train_files'],
                                                               pool['train_labels'],
                                                               len(pool['class_names']), model_info, is_augmented)

    val_features, val_labels = get_prediction_by_generator(model, image_dir, pool['val_files'],
                                                           pool['val_labels'],
                                                           len(pool['class_names']), model_info, is_augmented)

    test_features, test_labels = get_prediction_by_generator(model, image_dir, pool['test_files'],
                                                             pool['test_labels'],
                                                             len(pool['class_names']), model_info, is_augmented)

    return train_features, val_features, test_features, train_labels, val_labels, test_labels


def main():
    data_pools = load_pickle('/home/long/Desktop/Hela_split_30_2018-12-04.pickle')
    data, filepath = save_features_and_prediction('/home/long/Desktop', 'alexnet',
                                 '/home/long/finetune/saved_models/Hela_split_30_2018-12-04_0_alexnet',
                                 "/mnt/6B7855B538947C4E/Dataset/JPEG_data/Hela_JPEG", data_pools, '0', False)


if __name__ == "__main__":
    main()
