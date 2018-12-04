
import pickle
from datetime import datetime
from PIL import Image
import numpy as np
import os

def current_date(now):
    return now.strftime('%Y-%m-%d')

def current_time(now):
    return now.strftime('%Y-%m-%d_%H:%M:%S')


def dump_pickle(dict, path):
    filepath = path + '.pickle'
    with open(filepath, 'wb') as handle:
        pickle.dump(dict, handle)
    return filepath


def load_pickle(filepath):
    with open(filepath, 'rb') as handle:
        dict = pickle.load(handle)
    return dict

def read_image(path):
    image = Image.open(path)
    image.load()
    data = np.asarray(image, dtype="float32")
    return data

def prepare_numpy_data_arr(image_dir, image_short_path_arr):
    data_arr = []

    for image_short_path in image_short_path_arr:
        image_path = os.path.join(image_dir, image_short_path)
        image_data = read_image(image_path)
        data_arr.append(image_data)
    data_arr = np.asarray(data_arr)
    return data_arr

