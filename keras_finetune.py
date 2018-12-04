
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
from keras import optimizers, metrics, initializations
from keras.callbacks import EarlyStopping
from keras.models import load_model, model_from_json, Model
from keras import backend as K

from split_data import print_split_report
from utils import *
from data_generator import get_generators
from nets.googlenet import *
from nets.alexnet import *


def create_model_info(architecture):
    model_info = {}
    if architecture == 'alexnet':
        model_info['bottleneck_tensor_size'] = 2048
        model_info['input_width'] = 299
        model_info['input_height'] = 299
        model_info['input_depth'] = 3
        model_info['pretrained_weights'] = '/mnt/6B7855B538947C4E/pretrained_model/keras/resnet152_weights_tf.h5'

    elif architecture == 'googlenet':
        model_info['bottleneck_tensor_size'] = 1024
        model_info['input_width'] = 224
        model_info['input_height'] = 224
        model_info['input_depth'] = 3
        model_info['pretrained_weights'] = '/mnt/6B7855B538947C4E/pretrained_model/keras/resnet152_weights_tf.h5'

    else:
        raise Exception
    return model_info

def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)
def declare_model(num_classes, architecture, model_info, dropout=0):
    if architecture == 'alexnet':
        p_model, base_model= AlexNet(model_info['pretrained_weights'])
        print (p_model.summary())
        print(base_model.summary())

    elif architecture == 'googlenet':
        p_model, base_model = create_googlenet(model_info['pretrained_weights'])
        print(p_model.summary())
        print(base_model.summary())


    num_base_layers = len(base_model.layers)

    input = base_model.input
    x = base_model.output

    x = Dropout(dropout)(x)
    predictions = Dense(num_classes, input_shape=(model_info['bottleneck_tensor_size'],), activation='softmax')(x)
    model = Model(input=input, outputs=predictions)

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


# def get_np_data(split, image_dir):
#     train_images = split['train_files']
#     train_labels = split['train_labels']
#
#     val_images = split['val_files']
#     val_labels = split['val_labels']
#
#     test_images = split['test_files']
#     test_labels = split['test_labels']
#     num_classes = len(split['class_names'])
#
#     train_data = prepare_numpy_data_arr(image_dir, train_images)
#     val_data = prepare_numpy_data_arr(image_dir, val_images)
#     test_data = prepare_numpy_data_arr(image_dir, test_images)
#
#     return train_data, np.asarray(train_labels), val_data, np.asarray(val_labels), test_data, np.asarray(test_labels)


# TODO: save train log, return performance result
# TODO: retrain some layers with small learning rate after finetuning -> can do it later by restoring and train few last layers
# TODO: export to pb file
# return val_score, test_score in dict form: test_score = {'acc': model accuracy, 'loss', model loss}
def train(split, image_dir, architecture, hyper_params, log_path=None, save_model_path=None, restore_model_path=None,
          train_batch=8, test_batch=16, num_last_layer_to_finetune=-1):

    model_info = create_model_info(architecture)

    train_generator, validation_generator, test_generator = get_generators(split, image_dir, train_batch,
                                                                           test_batch)
    num_classes = len(split['class_names'])
    train_len = len(split['train_files'])
    validation_len = len(split['val_files'])
    test_len = len(split['test_files'])

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
    # optimizer = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.0, nesterov=False)  # Inception-Resnet
    # optimizer = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.99)
    # optimizer = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

    # TODO: fix that
    model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                  metrics=['accuracy'])  # cal accuracy and loss of the model; result will be a dict

    '''
    Train the model 
    '''
    # note that keras 2 have problems with sample_per_epochs -> need to use sample per epoch
    # see https://github.com/keras-team/keras/issues/5818
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0,
                                   mode='auto')
    # save tensorboard log if logdir is not None


    model.fit_generator(
        train_generator,
        epochs=100,
        steps_per_epoch=train_len // train_batch + 1,
        validation_data=validation_generator,
        validation_steps=validation_len // test_batch + 1,
        callbacks=[early_stopping],
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

    # save the model if dir is passed
    if save_model_path is not None:
        save_model(model, save_model_path)

    return train_score, val_score, test_score


def save_model(model, path):
    # serialize model to JSON
    model_json = model.to_json()
    open(path + '.json', "x")  # create the file
    with open(path + '.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path + '.h5')
    print("Saved model to disk")
    # try:
    #     model.save(path+'.h5')
    # except:
    #     print('cannot save model')
    #     pass
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


def main(_):
    '''
    prepare data
    '''
    data_pools = load_pickle('/home/long/Desktop/Hela_split_30_2018-12-04.pickle')
    pool = data_pools['data']['0']
    print(pool['data_name'])
    print(len(pool['train_files']))
    print_split_report('train', pool['train_report'])
    num_classes = len(pool['class_names'])

    # '''
    # Test train
    # '''
    #
    train_score, val_score, test_score = train(pool, '/mnt/6B7855B538947C4E/Dataset/JPEG_data/Hela_JPEG', 'alexnet',
          {'lr': 0.1, 'lr_decay': 0, 'momentum': 0,  'nesterov': False}, save_model_path='/home/long/Desktop/keras_alexnet', train_batch=8, test_batch=16)


    # '''
    # Test restore and eval
    # '''
    #
    # hyper_params = {'lr': 0.2, 'lr_decay': 0, 'momentum': 0, 'nesterov': False}
    # model_info = create_model_info('resnet_v2')
    #
    # # model, _ = restore_model('/home/ndlong95/finetune/saved_models/Hela_split_30_2018-07-19_0_resnet_v2', hyper_params)
    # model, _ = get(model_info, num_classes)
    # model.load_weights('/home/long/Desktop/Hela_split_30_2018-07-19_0_resnet_v2.h5', by_name=True)
    #
    # train_generator, validation_generator, test_generator = get_generators(model_info, pool,
    #                                                                        '/mnt/6B7855B538947C4E/Dataset/JPEG_data/Hela_JPEG',
    #                                                                        8,
    #                                                                        16)
    # train_len = len(pool['train_files'])
    # validation_len = len(pool['val_files'])
    # test_len = len(pool['test_files'])
    # train_score = model.evaluate_generator(train_generator, train_len // 8 + 1)
    # train_score = {'loss': train_score[0], 'acc': train_score[1]}
    # print('train_score: ', train_score)
    #
    # val_score = model.evaluate_generator(validation_generator, validation_len // 16 + 1)
    # val_score = {'loss': val_score[0], 'acc': val_score[1]}
    # print('val_score: ', val_score)
    #
    # test_score = model.evaluate_generator(test_generator, test_len // 16 + 1)
    # test_score = {'loss': test_score[0], 'acc': test_score[1]}
    # print('test score: ', test_score)


if __name__ == '__main__':
    main()
