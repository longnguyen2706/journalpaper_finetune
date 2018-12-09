
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

def read_images(image_dir, image_short_path_arr):
    data = []
    for image_short_path in image_short_path_arr:
        image_path = os.path.join(image_dir, image_short_path)
        image_data = read_image(image_path)

        # if gray image -> stack to make 3 dim
        if (len(image_data.shape) == 2):
            image_data = np.stack((image_data,) * 3, axis=-1)
        data.append(image_data)
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

def crop(image_data, x, y, w, h):
    crop_img = image_data[y:y + h, x:x + w]
    return crop_img

def crop_on_position(image_data, h, w, position):
    (height, width, _) = image_data.shape

    if (position == 'top_left'):
        return crop(image_data, 0, 0, w, h)
    elif (position == 'bottom_left'):
        return crop(image_data, 0, height-h, w, h)
    elif(position == 'top_right'):
        return crop(image_data, width-w, 0, w, h)
    elif(position == 'bottom_right'):
        return crop(image_data, width-w, height-h, w, h)
    elif(position == 'center'):
        return crop (image_data, int((width-w)/2) , int((height-h)/2), w, h)
    else:
        pass

def flip_x_axis(image_data):
    return np.flipud(image_data)

def crop_all_position_and_flip(image_data, w, h):
    data = []
    top_left = crop_on_position(image_data, h, w, 'top_left')
    data.append(top_left)
    data.append(flip_x_axis(top_left))

    bottom_left = crop_on_position(image_data, h, w, 'bottom_left')
    data.append(bottom_left)
    data.append(flip_x_axis(bottom_left))

    top_right = crop_on_position(image_data, h, w, 'top_right')
    data.append(top_right)
    data.append(flip_x_axis(top_right))

    bottom_right = crop_on_position(image_data, h, w, 'bottom_right')
    data.append(bottom_right)
    data.append(flip_x_axis(bottom_right))

    center = crop_on_position(image_data, h, w, 'center')
    data.append(center)
    data.append(flip_x_axis(center))

    data = np.asarray(data)
    # print ("shape of all crop and flip array: ", data.shape)
    return data


def prepare_image_data_arr_and_label(image_dir, image_short_path_arr, height, width, mean, std, label_arr):
    data_arr = read_images(image_dir, image_short_path_arr)

    for image_data in data_arr:
        # do resize
        image_data = resize_image(image_data, width, height)

        # do normalize
        # image_data = normalize_image(image_data, mean, std)

        # reorder dim from (height, width, channel) -> (channel, height, width) to suit theano
        transposed_data = image_data.transpose(2, 0, 1)
        data_arr.append(transposed_data)

    return np.asarray(data_arr), np.asarray(label_arr)

def prepare_augmented_data_and_label(image_dir, image_short_path_arr, height, width, std, mean, label_arr):
    aug_data_arr = []
    aug_label_arr = []

    data_arr = read_images(image_dir, image_short_path_arr)

    for i in range(0, len(data_arr)):
        data = data_arr[i]
        label = label_arr[i]
        aug_data = crop_all_position_and_flip(data, height, width)
        for image_data in aug_data:
            # do normalize
            # image_data = normalize_image(image_data, mean, std)

            # reorder dim from (height, width, channel) -> (channel, height, width) to suit theano
            transposed_data = image_data.transpose(2,0,1)
            aug_data_arr.append(transposed_data)

        for j in range(0, len(aug_data)):
            aug_label_arr.append(label)

    return np.asarray(aug_data_arr), np.asarray(aug_label_arr)

if __name__ == "__main__":
    # this one is just to test whether the functions working fine
    import matplotlib.pyplot as plt
    data = read_image('/home/long/Desktop/skater.jpg')
    # plt.subplot(3,1,1)
    # plt.imshow(np.asarray(data, dtype='uint8'))
    #
    # flip_data = flip_x_axis(data)
    # flip_data = np.asarray(flip_data, dtype="uint8")
    # plt.subplot(3, 1, 2)
    # plt.imshow(flip_data)
    #
    # crop_image = crop_on_position(data, 501, 501, "bottom_right")
    # crop_image= np.asarray(crop_image, dtype="uint8")
    # print(crop_image.shape)
    # plt.subplot(3, 1, 3)
    # plt.imshow(crop_image)
    # plt.show()

    crop_all_position_and_flip(data, 500, 500)