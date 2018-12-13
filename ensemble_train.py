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

from svm_classifier import SVM_CLASSIFIER
from utils import *

IMAGE_DIR = '/mnt/6B7855B538947C4E/Dataset/JPEG_data/Hela_JPEG'
OUT_MODEL1 = '/mnt/6B7855B538947C4E/home/duclong002/handcraft_models/stage1.pkl'

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
DIM_REDUCER = PCA(n_components=300, whiten=True, random_state=42,svd_solver='randomized')


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

def train_and_eval_svm(data):

    param_grid = gen_grid(HYPER_PARAMS)
    cls1 = SVM_CLASSIFIER(param_grid, CLASSIFIER1, OUT_MODEL1)
    cls1.prepare_model()
    cls1.train(reshape_2D(data['train_features']), argmax_label(reshape_2D(data['train_labels'])))
    print("Finish train svm")

    print("Now eval svm on val set")
    cls1_val = cls1.test(reshape_2D(data['val_features']), argmax_label(reshape_2D(data['val_labels'])), data['class_names'])
    acc_val_svm = cls1_val['accuracy']


    print("Now eval stage 1 on test set")
    cls1_test = cls1.test(reshape_2D(data['test_features']), argmax_label(reshape_2D(data['test_labels'])), data['class_names'])
    acc_test_svm = cls1_test['accuracy']

    print("---------------------")

    return acc_val_svm, acc_test_svm

def main():
    all_acc_val_svm = []
    all_acc_test_svm = []

    data = load_pickle("/home/long/Desktop/Hela_0_2018-12-04_0_alexnet.pickle")
    acc_val_svm, acc_test_svm = train_and_eval_svm(data)

    all_acc_val_svm.append(acc_val_svm)
    all_acc_test_svm.append(acc_test_svm)


if __name__ == "__main__":
    main()
