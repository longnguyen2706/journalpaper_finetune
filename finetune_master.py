import argparse

import os

import datetime

import sys

from extract_features import save_features_and_prediction
from keras_finetune import train_by_fit, train_by_fit_generator
import numpy as np

from split_data import print_split_report
from utils import *

sgd_hyper_params = {
    'learning_rates':[0.01, 0.05, 0.005], # u can try different values and watch. the paper use 5e-6 so u may want to try
    'lr_decays': [0, 1e-6], #[0, 1e-3, 1e-6], # u can try different values here. The paper use 0
    'momentums':[0, 0.9], # u may try to set it either 0 or 0.9 (0.9 is what the paper used)
    'nesterovs' : [False] # left this one False first (we might consider using nesterov later)
}

FINAL_HYPER_PARAMS = {
    'lr': 0.01,
    'lr_decay': 1e-6,
    'momentum': 0.9,
    'nesterov': False
}

IS_FINAL = True

#TODO: flags - pickle dir, splits no to train_by_fit, image_dir
FLAGS = None

'''
Train a single pool with hyper tuning
The model will be trained multiple times with different params setting and record the result
The best params then chosen based on val acc. 
The model will be train_by_fit again using this params. Model will be saved as .h5 and .pb file. Tensorboard log also be saved
Returns:
    dict: results of all train_by_fit with different hyper params and the final train_by_fit result with best hyper params
'''
def train_single_pool(pool_split, image_dir, log_path, architecture, save_model_path, train_batch, test_batch, is_augmented):
    results = {}
    results['hyper_tuning_result'] = []
    print('architecture: ', architecture)
    # hyper tuning and record result
    for lr in sgd_hyper_params['learning_rates']:
        for lr_decay in sgd_hyper_params['lr_decays']:
            for momentum in sgd_hyper_params['momentums']:
                for nesterov in sgd_hyper_params['nesterovs']:
                    hyper_params = {'lr': lr, 'lr_decay': lr_decay, 'momentum': momentum,  'nesterov': nesterov }
                    train_score, val_score, test_score = train_by_fit(pool_split, image_dir, architecture, hyper_params, is_augmented,
                                                                                train_batch=train_batch, test_batch=test_batch)
                    result = {
                        'hyper_params': hyper_params,
                        'train_score': train_score,
                        'test_score': test_score,
                        'val_score': val_score
                    }
                    results['hyper_tuning_result'].append(result)

    # for debug
    print('all results: ', results)

    # choosing the best params
    val_accuracies = []
    for result in results['hyper_tuning_result']:
        val_accuracy = result['val_score']['acc']
        val_accuracies.append(val_accuracy)

    val_accuracies = np.asarray(val_accuracies)
    best_val_acc_index = np.argmax(val_accuracies)
    print ('best val acc: ', val_accuracies[best_val_acc_index])
    # for debug
    print ('best result: ', results['hyper_tuning_result'][best_val_acc_index])

    # retrain the model with the best params and save the model to .h5 and .pb
    best_hyper_params =results['hyper_tuning_result'][best_val_acc_index]['hyper_params']
    final_train_score, final_val_score, final_test_score = train_by_fit(pool_split, image_dir, architecture, hyper_params, is_augmented,
                                                                                  save_model_path= save_model_path, log_path=log_path,
                                                                                  train_batch=train_batch, test_batch=test_batch)
    final_result = {
        'hyper_params': best_hyper_params,
        'train_score': final_train_score,
        'test_score': final_test_score,
        'val_score': final_val_score
    }

    results['final_result']=final_result
    return results

def train_single_pool_final(pool_split, image_dir, log_path, architecture, save_model_path, train_batch, test_batch, is_augmented):
    results ={}

    final_train_score, final_val_score, final_test_score = train_by_fit(pool_split, image_dir, architecture,
                                                                        FINAL_HYPER_PARAMS, is_augmented,
                                                                        save_model_path=save_model_path,
                                                                        log_path=log_path,
                                                                        train_batch=train_batch, test_batch=test_batch)
    final_result = {
        'hyper_params': FINAL_HYPER_PARAMS,
        'train_score': final_train_score,
        'test_score': final_test_score,
        'val_score': final_val_score
    }

    results['final_result'] = final_result
    return results
'''
    train models with given pools and architecture
    record result to .pickle file 
'''
def train_pools(_):
    print(FLAGS)
    pools= load_pickle(FLAGS.pool_dir)
    start_pool_idx = int(FLAGS.start_pool)
    end_pool_idx = int(FLAGS.end_pool)

    now = datetime.datetime.now()
    time = current_time(now)

    if not os.path.exists(FLAGS.save_model_dir):
        os.makedirs(FLAGS.save_model_dir)
    if not os.path.exists (FLAGS.result_dir):
        os.makedirs(FLAGS.result_dir)
    trained_models_info = []

    all_results = []

    for idx in range(start_pool_idx, end_pool_idx+1):
        pool = pools['data'][str(idx)]
        print ('pool idx: ', idx)
        print ('****************')
        print_split_report('train', pool['train_report'])
        print_split_report('val', pool['val_report'])
        print_split_report('test', pool['test_report'])
        print('-----------------')

        name = pools['pool_name']+'_'+str(idx)
        log_path = os.path.join(FLAGS.log_dir, name, FLAGS.architecture)
        save_model_path = os.path.join(FLAGS.save_model_dir, name+'_'+str(FLAGS.architecture))
        if not IS_FINAL:
            results = train_single_pool(pool, FLAGS.image_dir, log_path, FLAGS.architecture,
                          save_model_path, FLAGS.train_batch, FLAGS.test_batch, FLAGS.is_augmented)
        else:
            results = train_single_pool_final(pool, FLAGS.image_dir, log_path, FLAGS.architecture,
                              save_model_path, FLAGS.train_batch, FLAGS.test_batch, FLAGS.is_augmented)
            save_features_and_prediction(FLAGS.result_dir, FLAGS.architecture,
                                         save_model_path,
                                         FLAGS.image_dir, pools, str(idx), False)
        model_info = {
            'hyper_param_setting':sgd_hyper_params,
            'pool_idx': str(idx),
            'pool_name': pool['data_name'],
            'time': time,
            'architecture': FLAGS.architecture,
            'train_batch': FLAGS.train_batch,
            'test_batch': FLAGS.test_batch,
            'log_path': log_path,
            'save_model_path': save_model_path,
            'results': results,
            'final_results': results['final_result']
        }
        trained_models_info.append(model_info)
        all_results.append(results['final_result']['test_score'])

    # save result to .pickle
    trained_models_info_pickle_name = pools['pool_name']+'_'+str(start_pool_idx)+'_'+str(end_pool_idx)
    dump_pickle(trained_models_info, os.path.join(FLAGS.result_dir, trained_models_info_pickle_name))

    cal_mean_and_std(all_results, "avg test acc")
    return trained_models_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pool_dir',
        type=str,
    )

    parser.add_argument(
        '--image_dir',
        type=str,
    )

    parser.add_argument(
        '--architecture',
        type=str
    )

    parser.add_argument(
        '--start_pool',
        type=int
    )

    parser.add_argument(
        '--end_pool',
        type=int
    )

    parser.add_argument(
        '--log_dir',
        type=str,
    )
    parser.add_argument(
        '--save_model_dir',
        type=str,
    )
    parser.add_argument(
        '--result_dir',
        type=str,
    )

    parser.add_argument(
        '--train_batch',
        default=8,
        type=int
    )
    parser.add_argument(
        '--test_batch',
        default=16,
        type=int
    )
    parser.add_argument(
        '--is_augmented',
        default=False,
        type= bool
    )
    FLAGS, unparsed = parser.parse_known_args()
    train_pools([sys.argv[0]] + unparsed)

