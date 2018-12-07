from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras import optimizers, metrics, initializations
from keras.callbacks import EarlyStopping
from keras.models import load_model, model_from_json, Model
from keras import backend as K

from split_data import print_split_report
from utils import *
from nets.googlenet import *
from nets.alexnet import *
from keras.utils import np_utils


def create_model_info(architecture):
    model_info = {}
    if architecture == 'alexnet':
        model_info['bottleneck_tensor_size'] = 4096
        model_info['input_width'] = 227
        model_info['input_height'] = 227
        model_info['input_depth'] = 3
        model_info['input_mean'] = 128
        model_info['input_std'] = 128
        model_info['pretrained_weights'] =  "/mnt/6B7855B538947C4E/deeplearning/pretrained_weights/alexnet_weights.h5"

    elif architecture == 'googlenet':
        model_info['bottleneck_tensor_size'] = 1024
        model_info['input_width'] = 224
        model_info['input_height'] = 224
        model_info['input_depth'] = 3
        model_info['input_mean'] = 128
        model_info['input_std'] = 128
        model_info['pretrained_weights'] = '/mnt/6B7855B538947C4E/deeplearning/pretrained_weights/googlenet_weights.h5'

    else:
        raise Exception
    return model_info


def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)


def declare_model(num_classes, architecture, model_info, dropout=0):
    if architecture == 'alexnet':
        p_model, base_model = AlexNet(model_info['pretrained_weights'])
        print(p_model.summary())
        print(base_model.summary())

    elif architecture == 'googlenet':
        p_model, base_model = create_googlenet(model_info['pretrained_weights'])
        print(p_model.summary())
        print(base_model.summary())

    num_base_layers = len(base_model.layers)

    input = base_model.input
    x = base_model.output

    x = Dropout(dropout)(x)
    predictions = Dense(num_classes, input_dim=(model_info['bottleneck_tensor_size']), activation='softmax',
                        name='dense_finetune')(x)
    model = Model(input=input, output=predictions)

    return model, num_base_layers


def set_model_trainable(model, num_base_layers, num_of_last_layer_finetune):
    if num_of_last_layer_finetune == -1:  # retrain all layers
        for layer in model.layers[:num_base_layers]:
            layer.trainable = True

    elif num_of_last_layer_finetune <= num_base_layers:
        for layer in model.layers[:(num_base_layers - num_of_last_layer_finetune)]:
            layer.trainable = False
        for layer in model.layers[(num_base_layers - num_of_last_layer_finetune):]:
            layer.trainable = True

    print(model.summary())
    return model


def get_np_data(split, image_dir, model_info):
    train_images = split['train_files']
    train_labels = split['train_labels']

    val_images = split['val_files']
    val_labels = split['val_labels']

    test_images = split['test_files']
    test_labels = split['test_labels']
    num_classes = len(split['class_names'])

    train_data = prepare_numpy_data_arr(image_dir, train_images, model_info['input_height'],
                                        model_info['input_width'], model_info['input_mean'],
                                        model_info['input_std'])
    val_data = prepare_numpy_data_arr(image_dir, val_images, model_info['input_height'],
                                        model_info['input_width'], model_info['input_mean'],
                                        model_info['input_std'])
    test_data = prepare_numpy_data_arr(image_dir, test_images, model_info['input_height'],
                                        model_info['input_width'], model_info['input_mean'],
                                        model_info['input_std'])

    train_labels = np_utils.to_categorical(np.asarray(train_labels), num_classes)
    val_labels = np_utils.to_categorical(np.asarray(val_labels), num_classes)
    test_labels = np_utils.to_categorical(np.asarray(test_labels), num_classes)

    print('train data shape: ', train_data.shape)
    print('val data shape: ', val_data.shape)
    print('test data shape: ', test_data.shape)

    print('train label shape: ', train_labels.shape)
    print('val label shape: ', val_labels.shape)
    print('test label shape: ', test_labels.shape)

    return (train_data, np.asarray(train_labels)), (val_data, np.asarray(val_labels)), (
        test_data, np.asarray(test_labels))


# TODO: save train log, return performance result
# TODO: retrain some layers with small learning rate after finetuning -> can do it later by restoring and train few last layers
# TODO: export to pb file
# return val_score, test_score in dict form: test_score = {'acc': model accuracy, 'loss', model loss}
def train(pool, image_dir, architecture, hyper_params, log_path=None, save_model_path=None, restore_model_path=None,
          train_batch=16, test_batch=32, num_last_layer_to_finetune=-1):
    model_info = create_model_info(architecture)
    print(pool['data_name'])
    print(len(pool['train_files']))
    print_split_report('train', pool['train_report'])
    num_classes = len(pool['class_names'])

    # train the model from scratch or train the model from some point
    if restore_model_path == None:
        model, num_base_layers = declare_model(num_classes, architecture, model_info)
        model = set_model_trainable(model, num_base_layers, num_last_layer_to_finetune)
    else:
        model, num_layers = restore_model(restore_model_path, hyper_params)
        model = set_model_trainable(model, num_layers, num_last_layer_to_finetune)

    print('training the model with hyper params: ', hyper_params)
    optimizer = optimizers.SGD(lr=hyper_params['lr'], decay=hyper_params['lr_decay'],
                               momentum=hyper_params['momentum'], nesterov=hyper_params['nesterov'])  # Inception

    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = get_np_data(pool,image_dir,model_info)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0,
                                   mode='auto')
    # # TODO: fix that
    model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                  metrics=['accuracy'])  # cal accuracy and loss of the model; result will be a dict

    model.fit(X_test, Y_test,
              batch_size=train_batch,
              nb_epoch=1,
              shuffle=True,
              verbose=1,
              validation_data=(X_val, Y_val),
              callbacks=[early_stopping]
              )

    train_score = model.evaluate(X_train, Y_train, test_batch)
    train_score = {'loss': train_score[0], 'acc': train_score[1]}
    print('train_score: ', train_score)

    val_score =  model.evaluate(X_val, Y_val, test_batch)
    val_score = {'loss': val_score[0], 'acc': val_score[1]}
    print('val_score: ', val_score)

    test_score = model.evaluate(X_test, Y_test, test_batch)
    test_score = {'loss': test_score[0], 'acc': test_score[1]}
    print('test score: ', test_score)
    if save_model_path is not None:
        save_model(model, save_model_path)

    return train_score, val_score, test_score


def save_model(model, path):
    model.save_weights(path + '.h5')
    print("Saved model to disk")
    return


def restore_model(model_path, hyper_params):
    # model = load_model(model_path)

    # load json and create model
    json_file = open(model_path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_path + '.h5')
    print("Loaded model from disk")
    num_layers = len(model.layers)

    # compile model with appropriate setting
    print('restore the model with hyper params: ', hyper_params)
    optimizer = optimizers.SGD(lr=hyper_params['lr'], decay=hyper_params['lr_decay'],
                               momentum=hyper_params['momentum'], nesterov=hyper_params['nesterov'])

    model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                  metrics=['accuracy'])

    print('Restored model from path ', model_path)
    print(model.summary())
    return model, num_layers


def _try():
    architecture = 'alexnet'
    model_info = create_model_info(architecture)

    data_pools = load_pickle('/home/duclong002/Desktop/Hela_split_30_2018-12-07.pickle')
    pool = data_pools['data']['0']
    print(pool['data_name'])
    print(len(pool['train_files']))
    print_split_report('train', pool['train_report'])
    num_classes = len(pool['class_names'])

    #
    model, num_base_layers = declare_model(num_classes, architecture, model_info)
    model = set_model_trainable(model, num_base_layers, -1)

    # img_rows, img_cols = 224, 224 # Resolution of inputs
    # channel = 3
    # num_classes = 10
    batch_size = 16
    nb_epoch = 50
    # X_train, Y_train, X_val, Y_val = load_cifar10_data(img_rows, img_cols)
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = get_np_data(pool,
                                                                       "/home/duclong002/Dataset/JPEG_data/Hela_JPEG",
                                                                       model_info)
    optimizer = optimizers.SGD(lr=0.01, decay=1e-6)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0,
                                   mode='auto')
    # # TODO: fix that
    model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                  metrics=['accuracy'])  # cal accuracy and loss of the model; result will be a dict


    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_val, Y_val),
              callbacks=[early_stopping]
              )

    test_score = model.evaluate(X_test, Y_test, 32)
    test_score = {'loss': test_score[0], 'acc': test_score[1]}
    print('test score: ', test_score)

def main():
    _try()


if __name__ == '__main__':
    main()
