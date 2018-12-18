from __future__ import absolute_import
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import copy
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.utils import Bunch
import collections

import re
from svm_classifier import SVM_CLASSIFIER
from utils import *

OUT_MODEL1 = '/home/duclong002/handcraft_models/stage1.pkl'
OUT_MODEL2 = 'home/duclong002/handcraft_models/stage2.pkl'
OUT_MODEL3 = '/home/duclong002/handcraft_models/stage3.pkl'

FEATURE_DIR = "/home/duclong002/journal_paper_finetune/archived_results/"
PCA_PERCENTAGE = 95
# PARAM_GRID = {'linearsvc__C': [1, 5, 10, 50]}

HYPER_PARAMS = [
    {
        'pow_min': -15,
        'pow_max': 15,
        'base': 2,
        'pow_step': 1,
        'type': 'linearsvc__C',
    }
]
CLASSIFIER1 = svm.LinearSVC()

def gen_grid(hyper_params):
    params_grid ={}
    for hyper_param in hyper_params:
        grid_params = []
        for i in range(hyper_param['pow_max'] - hyper_param['pow_min'] + 1):
            if (i % hyper_param['pow_step'] == 0):
                grid_params.append(pow(hyper_param['base'], hyper_param['pow_min'] + i))
        params_grid[str(hyper_param['type'])]=grid_params
    print('param grids for HYPER PARAMS: ', hyper_params, params_grid)
    return params_grid

def reshape_2D(arr):
    if len(arr.shape) ==3:
        x, y, z = arr.shape
    reshaped_arr = np.reshape(arr, (x*y,z))
    print ("original and new shape: ", arr.shape, reshaped_arr.shape)
    return reshaped_arr

def argmax_label(encoded_data):
    decoded_data = []
    for i in range(encoded_data.shape[0]):
        decoded_datum = np.argmax(encoded_data[i])
        decoded_data.append(decoded_datum)
    return np.asarray(decoded_data)

def eval_finetune(data):
    val_prediction = argmax_label(reshape_2D(data['val_prediction']))
    val_labels = argmax_label(reshape_2D(data['val_labels']))
    val_accuracy = accuracy_score(val_labels, val_prediction)

    test_prediction = argmax_label(reshape_2D(data['test_prediction']))
    test_labels = argmax_label(reshape_2D(data['test_labels']))
    test_accuracy = accuracy_score(test_labels, test_prediction)

    return val_accuracy, test_accuracy


def eval_ensemble(categorical_prediction_arr, categorial_labels):
    ensemble_categorical_prediction = np.sum(categorical_prediction_arr, axis=0)
    ensemble_prediction = argmax_label(ensemble_categorical_prediction)
    print(ensemble_prediction.shape)

    acc = accuracy_score(argmax_label(categorial_labels), ensemble_prediction)
    print ("esemble acc: ", acc)
    return acc

def concat(feature_arr1, feature_arr2):
    feature_concat = []
    for i in range(0, len(feature_arr1)):
        feature = np.concatenate((feature_arr1[i], feature_arr2[i]))
        assert(feature.shape == ((4096+1024),))
        feature_concat.append(feature)

    return np.asarray(feature_concat)

def train_and_eval_single_concat(data1, data2, pca_percentage):
    pca = get_PCA(pca_percentage)

    train_features = concat(reshape_2D(data1['train_features']), reshape_2D(data2['train_features']))
    val_features = concat(reshape_2D(data1['val_features']), reshape_2D(data2['val_features']))
    test_features = concat(reshape_2D(data1['test_features']), reshape_2D(data2['test_features']))

    pca.fit(train_features)
    print(pca.n_components_)

    param_grid = gen_grid(HYPER_PARAMS)
    cls1 = SVM_CLASSIFIER(param_grid, CLASSIFIER1, OUT_MODEL3, pca)
    cls1.prepare_model()
    cls1.train(train_features, argmax_label(reshape_2D(data1['train_labels'])))
    print("Finish train concat svm")

    print("Now eval concat svm on val set")
    cls1_val = cls1.test(val_features, argmax_label(reshape_2D(data1['val_labels'])),
                         data1['class_names'])

    print("Now eval concat svm on test set")
    cls1_test = cls1.test(test_features, argmax_label(reshape_2D(data1['test_labels'])),
                          data1['class_names'])
    print("---------------------")

    return cls1_val['accuracy'], cls1_test['accuracy'], cls1_val['prediction'], cls1_test['prediction']

def train_and_eval_svm(data, pca_percentage):

    pca = get_PCA(pca_percentage)
    pca.fit(reshape_2D(data['train_features']))
    print(pca.n_components_)

    param_grid = gen_grid(HYPER_PARAMS)
    cls1 = SVM_CLASSIFIER(param_grid, CLASSIFIER1, OUT_MODEL1, pca)
    cls1.prepare_model()
    cls1.train(reshape_2D(data['train_features']), argmax_label(reshape_2D(data['train_labels'])))
    print("Finish train svm")

    print("Now eval svm on val set")
    cls1_val = cls1.test(reshape_2D(data['val_features']), argmax_label(reshape_2D(data['val_labels'])), data['class_names'])

    print("Now eval stage 1 on test set")
    cls1_test = cls1.test(reshape_2D(data['test_features']), argmax_label(reshape_2D(data['test_labels'])), data['class_names'])
    print("---------------------")

    return cls1_val['accuracy'], cls1_test['accuracy'], cls1_val['prediction'], cls1_test['prediction']

def get_PCA(percentage):
    return PCA(n_components=percentage/100, random_state=42, svd_solver='full')

def find_all_pickles(dir, architecture):
    file_paths = []
    for path in os.listdir(dir):
        if (path.endswith(architecture + ".pickle")):
            file_paths.append(os.path.join(dir, path))
    ordered_files = sorted(file_paths, key=lambda x: (int(re.sub('\D', '', x)), x))
    print(ordered_files)
    return ordered_files

def train_and_eval_concats(dir, architecture1, architecture2, pca_percentage):
    all_files1 = find_all_pickles(dir, architecture1)
    all_files2 = find_all_pickles(dir, architecture2)
    all_p_svm_test = []
    all_label_test = []
    all_acc_val_svm = []
    all_acc_test_svm = []
    for i in range(0, len(all_files1)):
        data1 = load_pickle(all_files1[i])
        data2 = load_pickle(all_files2[i])
        acc_val_svm, acc_test_svm, p_val_svm, p_test_svm = train_and_eval_single_concat(data1, data2, pca_percentage)

        all_acc_val_svm.append(acc_val_svm)
        all_acc_test_svm.append(acc_test_svm)
        all_p_svm_test.append( np_utils.to_categorical(p_test_svm))
        all_label_test.append(reshape_2D(data1['test_labels']))

    cal_mean_and_std(all_acc_val_svm, "val_svm")
    cal_mean_and_std(all_acc_val_svm, "test_svm")

    return all_p_svm_test, all_label_test

def avg_svm_finetune_ensemble(dir, architecture, pca_percentage):

    all_acc_val_svm1 = []
    all_acc_test_svm1 = []

    all_acc_val_finetune1 = []
    all_acc_test_finetune1 = []

    all_acc_val_ensemble1 = []
    all_acc_test_ensemble1 = []

    all_p_svm_test = []
    all_p_finetune_test = []

    all_files = find_all_pickles(dir, architecture)
    for file in all_files:
        data = load_pickle(file)
        acc_val_finetune, acc_test_finetune = eval_finetune(data)
        all_acc_val_finetune1.append(acc_val_finetune)
        all_acc_test_finetune1.append(acc_test_finetune)

        acc_val_svm, acc_test_svm, prediction_val_svm, prediction_test_svm = train_and_eval_svm(data, pca_percentage)
        all_acc_val_svm1.append(acc_val_svm)
        all_acc_test_svm1.append(acc_test_svm)

        all_val_prediction = [reshape_2D(data['val_prediction']), np_utils.to_categorical(prediction_val_svm)]
        val_ensemble = eval_ensemble(all_val_prediction, reshape_2D(data['val_labels']))
        all_acc_val_ensemble1.append(val_ensemble)

        all_test_prediction = [reshape_2D(data['test_prediction']), np_utils.to_categorical(prediction_test_svm)]
        test_ensemble = eval_ensemble(all_test_prediction, reshape_2D(data['test_labels']))
        all_acc_test_ensemble1.append(test_ensemble)

        all_p_svm_test.append(np_utils.to_categorical(prediction_test_svm))
        all_p_finetune_test.append(reshape_2D(data['test_prediction']))
    cal_mean_and_std(all_acc_val_svm1, "val_svm")
    cal_mean_and_std(all_acc_val_svm1, "test_svm")

    cal_mean_and_std(all_acc_val_finetune1, "val_finetune")
    cal_mean_and_std(all_acc_test_finetune1, "test_finetune")

    cal_mean_and_std(all_acc_val_ensemble1, "val_ensemble")
    cal_mean_and_std(all_acc_test_ensemble1, "test_ensemble")

    return all_p_svm_test, all_p_finetune_test, all_acc_test_svm1, all_acc_val_finetune1, all_acc_test_ensemble1


def main():
    # print ("----------------alexnet---------------------")
    # all_p_svm_test1, all_p_ft_test1, a_svm_test1, a_ft_test1, a_ensbl_test1 = avg_svm_finetune_ensemble(FEATURE_DIR, 'alexnet', PCA_PERCENTAGE)
    # print("---------------------------------------------")
    #
    # print("----------------googlenet--------------------")
    # all_p_svm_test2, all_p_ft_test2, a_svm_test2, a_ft_test2, a_ensbl_test2 = avg_svm_finetune_ensemble(FEATURE_DIR, 'googlenet', PCA_PERCENTAGE)
    # print("---------------------------------------------")

    print("----------------concat--------------------")
    all_p_svm_test_c, all_label_test = train_and_eval_concats(FEATURE_DIR, 'googlenet', 'alexnet', PCA_PERCENTAGE)
    print("---------------------------------------------")
    #
    # print("----------------ensemble--------------------")
    # all_acc_test_e = []
    # for i in range(0, all_p_ft_test1):
    #     pred = [all_p_ft_test1[i], all_p_ft_test2[i], all_p_svm_test1[i], all_p_svm_test2[i], all_p_svm_test_c[i]]
    #     acc_test_e = eval_ensemble(pred, all_label_test[i])
    #     all_acc_test_e.append(acc_test_e)
    #
    # cal_mean_and_std(all_acc_test_e, "test_ensemble")




if __name__ == "__main__":
    main()
