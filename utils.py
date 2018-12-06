
import pickle
from datetime import datetime
from PIL import Image
import numpy as np
import os
import cv2

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

'''
image_data: (height, width, channel)
'''
def resize_image(image_data, height, width):
    return cv2.resize(image_data, (height, width))

def normalize_image(image_data, mean, std):
    offset_image = image_data-mean
    mul_image = offset_image *(1.0/std)
    return mul_image

def prepare_numpy_data_arr(image_dir, image_short_path_arr, height, width, mean, std):
    data_arr = []

    for image_short_path in image_short_path_arr:
        image_path = os.path.join(image_dir, image_short_path)
        image_data = read_image(image_path)

        # if gray image -> stack to make 3 dim
        if (len(image_data.shape) == 2):
            image_data = np.stack((image_data,)*3, axis=-1)

        # do resize
        image_data = resize_image(image_data, width, height)

        # do normalize
        # image_data = normalize_image(image_data, mean, std)

        # reorder dim from (height, width, channel) -> (channel, height, width) to suit theano
        transposed_data = image_data.transpose(2, 0, 1)
        data_arr.append(transposed_data)

    data_arr = np.asarray(data_arr)
    return data_arr

