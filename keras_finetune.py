from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import optimizers, initializations
from keras.callbacks import EarlyStopping
from nets.alexnet import *
from nets.googlenet import *
from split_data import print_split_report
from utils import *
from utils import get_np_data


def create_model_info(architecture):
    model_info = {}
    if architecture == 'alexnet':
        model_info['bottleneck_tensor_size'] = 4096
        model_info['input_width'] = 227
        model_info['input_height'] = 227
        model_info['input_depth'] = 3
        model_info['input_mean'] = 128
        model_info['input_std'] = 128
        model_info['pretrained_weights'] = "/home/ndlong95/pretrained_model/keras/alexnet_weights.h5"

    elif architecture == 'googlenet':
        model_info['bottleneck_tensor_size'] = 1024
        model_info['input_width'] = 224
        model_info['input_height'] = 224
        model_info['input_depth'] = 3
        model_info['input_mean'] = 128
        model_info['input_std'] = 128
        model_info['pretrained_weights'] = '/home/ndlong95/pretrained_model/keras/googlenet_weights.h5'

    else:
        raise Exception
    return model_info


def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)


def declare_model(num_classes, architecture, model_info, dropout=0.5):
    if architecture == 'alexnet':
        p_model, base_model = AlexNet(model_info['pretrained_weights'])
        # print(p_model.summary())
        # print(base_model.summary())

    elif architecture == 'googlenet':
        p_model, base_model = create_googlenet(model_info['pretrained_weights'])
        # print(p_model.summary())
        # print(base_model.summary())

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

    # print(model.summary())
    return model

# TODO: save train log, return performance result
# TODO: retrain some layers with small learning rate after finetuning -> can do it later by restoring and train few last layers
# TODO: export to pb file
# return val_score, test_score in dict form: test_score = {'acc': model accuracy, 'loss', model loss}
def train_by_fit_generator(pool, image_dir, architecture, hyper_params, is_augmented, log_path=None, save_model_path=None,
                           restore_model_path=None,
                           train_batch=16, test_batch=32, num_last_layer_to_finetune=-1):
    model_info = create_model_info(architecture)
    print(pool['data_name'])
    print(len(pool['train_files']))
    print_split_report('train', pool['train_report'])
    num_classes = len(pool['class_names'])
    train_len = len(pool['train_files'])
    validation_len = len(pool['val_files'])
    test_len = len(pool['test_files'])
    print("train val, test len: ", train_len, validation_len, test_len)


    # trainr the model from scratch or train the model from some point
    if restore_model_path == None:
        print("training from scratch")
        model, num_base_layers = declare_model(num_classes, architecture, model_info)
        model = set_model_trainable(model, num_base_layers, num_last_layer_to_finetune)
    else:
        print("restoring model to train")
        model, _ , num_layers = restore_model_weight(architecture, num_classes, restore_model_path, False, hyper_params)
        model = set_model_trainable(model, num_layers, num_last_layer_to_finetune)

    print('training the model with hyper params: ', hyper_params)
    optimizer = optimizers.SGD(lr=hyper_params['lr'], decay=hyper_params['lr_decay'],
                               momentum=hyper_params['momentum'], nesterov=hyper_params['nesterov'])  # Inception
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0,
                                   mode='auto')
    model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                  metrics=['accuracy'])  # cal accuracy and loss of the model; result will be a dict

    train_generator = ThreadSafeGenerator(model_info, image_dir, pool['train_files'], pool['train_labels'], train_batch, num_classes, is_augmented)

    validation_generator = ThreadSafeGenerator(model_info, image_dir, pool['val_files'], pool['val_labels'], test_batch, num_classes, False)

    test_generator = ThreadSafeGenerator(model_info, image_dir, pool['test_files'], pool['test_labels'], test_batch, num_classes, False)

    model.fit_generator(
        train_generator,
        nb_epoch=100,
        samples_per_epoch=train_len // train_batch + 1,
        validation_data=validation_generator,
        nb_val_samples=validation_len // test_batch + 1,
        callbacks=[]
    )

    train_score = model.evaluate_generator(train_generator, train_len // train_batch + 1)
    train_score = {'loss': train_score[0], 'acc': train_score[1]}
    print('train_score: ', train_score)

    val_score = model.evaluate_generator(validation_generator, validation_len // test_batch + 1)
    val_score = {'loss': val_score[0], 'acc': val_score[1]}
    print('val_score: ', val_score)

    test_score = model.evaluate_generator(test_generator, test_len // test_batch + 1)
    test_score = {'loss': test_score[0], 'acc': test_score[1]}
    print('test score: ', test_score)

    if save_model_path is not None:
        save_model_weight(model, save_model_path)

    return train_score, val_score, test_score


def train_by_fit(pool, image_dir, architecture, hyper_params, is_augmented, log_path=None, save_model_path=None,
                           restore_model_path=None,
                           train_batch=16, test_batch=32, num_last_layer_to_finetune=-1):
    model_info = create_model_info(architecture)
    print(pool['data_name'])
    print(len(pool['train_files']))
    print_split_report('train', pool['train_report'])
    num_classes = len(pool['class_names'])
    train_len = len(pool['train_files'])
    validation_len = len(pool['val_files'])
    test_len = len(pool['test_files'])
    print("train, val, test len: ", train_len, validation_len, test_len)

    # train the model from scratch or train_by_fit_generator the model from some point
    if restore_model_path == None:
        print("training from scratch")
        model, num_base_layers = declare_model(num_classes, architecture, model_info)
        model = set_model_trainable(model, num_base_layers, num_last_layer_to_finetune)
    else:
        print("restoring model to train")
        model, _, num_layers = restore_model_weight(architecture, num_classes, restore_model_path, False, hyper_params)
        model = set_model_trainable(model, num_layers, num_last_layer_to_finetune)

    print('training the model with hyper params: ', hyper_params)
    optimizer = optimizers.SGD(lr=hyper_params['lr'], decay=hyper_params['lr_decay'],
                               momentum=hyper_params['momentum'], nesterov=hyper_params['nesterov'])  # Inception
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0,
                                   mode='auto')
    model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                  metrics=['accuracy'])  # cal accuracy and loss of the model; result will be a dict


    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = get_np_data(pool,
                                                                      image_dir,
                                                                       model_info, is_augmented)
    model.fit(X_train, Y_train,
              batch_size=train_batch,
              nb_epoch=50,
              shuffle=True,
              verbose=1,
              validation_data=(X_val, Y_val),
              callbacks=[early_stopping]
              )

    train_score = model.evaluate(X_train, Y_train, test_batch)
    train_score = {'loss': train_score[0], 'acc': train_score[1]}
    print('train score: ', train_score)

    val_score = model.evaluate(X_val, Y_val,test_batch)
    val_score = {'loss': val_score[0], 'acc': val_score[1]}
    print('val score: ', val_score)

    test_score = model.evaluate(X_test, Y_test,test_batch)
    test_score = {'loss': test_score[0], 'acc': test_score[1]}
    print('test score: ', test_score)
    if save_model_path is not None:
        save_model_weight(model, save_model_path)

    return train_score, val_score, test_score


def save_model_weight(model, path):
    model.save_weights(path + '.h5')
    print("Saved model to disk")


def restore_model_weight(architecture, num_classes, model_path, freeze = True, hyper_params=None):
    model_info = create_model_info(architecture)
    model, num_base_layers = declare_model(num_classes, architecture, model_info)

    # load weights into new model
    model.load_weights(model_path + '.h5')
    print("Loaded model from disk")

    num_layers = len(model.layers)

    if freeze:
        model = set_model_trainable(model, num_layers, 0)

    if hyper_params is not None:
        # compile model with appropriate setting
        print('restore the model with hyper params: ', hyper_params)

        optimizer = optimizers.SGD(lr=hyper_params['lr'], decay=hyper_params['lr_decay'],
                                   momentum=hyper_params['momentum'], nesterov=hyper_params['nesterov'])
        model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                      metrics=['accuracy'])

    print('Restored model from path ', model_path)
    print(model.summary())
    return model, num_base_layers, num_layers

def _try_fit():
    architecture = 'googlenet'
    model_info = create_model_info(architecture)

    data_pools = load_pickle('/home/duclong002/Desktop/Hela_split_30_2018-12-04.pickle')
    pool = data_pools['data']['0']
    print(pool['data_name'])
    print(len(pool['train_files']))
    print_split_report('train', pool['train_report'])
    num_classes = len(pool['class_names'])

    model, num_base_layers = declare_model(num_classes, architecture, model_info)
    model = set_model_trainable(model, num_base_layers, -1)

    batch_size = 16
    nb_epoch = 10


    is_augmented = False
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = get_np_data(pool,
                                                                       "/home/duclong002/Dataset/JPEG_data/Hela_JPEG",
                                                                       model_info, is_augmented)
    optimizer = optimizers.SGD(lr=0.01, decay=1e-6)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0,
                                   mode='auto')
    # # # TODO: fix that
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



def _try_generator():
    architecture = 'alexnet'
    model_info = create_model_info(architecture)
    data_pools = load_pickle('/home/duclong002/Desktop/Hela_split_30_2018-12-04.pickle')
    pool = data_pools['data']['0']

    train_generator = ThreadSafeGenerator(model_info, "/home/duclong002/Dataset/JPEG_data/Hela_JPEG",
                                     pool['train_files'], pool['train_labels'], 32, 10, False)
    for i in range(0, 100):
        batch_x, batch_y = next(train_generator)
        print(batch_x.shape)
        print(batch_y.shape)
        print("____________________________")


if __name__ == '__main__':
  _try_fit()
