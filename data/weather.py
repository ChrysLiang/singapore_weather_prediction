# CIFAR10 Downloader

import logging
import pickle
import math
import os
import errno
import tarfile
import shutil


import numpy as np

logger = logging.getLogger(__name__)

def get_train():
    return _get_dataset("train")

def get_test():

    return _get_dataset("test")

def get_shape_input():
    return (None, 30, 11, 9)

def get_shape_label():
    return (None,11*9)

def num_classes():
    return 10

def _unpickle_file(filename):
    logger.debug("Loading pickle file: {}".format(filename))

    with open(filename, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

     # Reorder the data
    img = data[b'data']
    img = img.reshape([-1, 3, 32, 32])
    img = img.transpose([0, 2, 3, 1])
    # Load labels
    lbl = np.array(data[b'labels'])

    return img, lbl


def _get_dataset(split):
    assert split == "test" or split == "train"
    path = "data"
    data_file = "weather_data_file_4_5.txt"

    file_path = os.path.join(path, data_file)

    # read the data_file line by line
    file = open(file_path, "r")
    content = file.read().split(sep="\n")

    imgs = []
    time_stamps = []
    targets = []
    # for each image
    nRows = 4
    nCols = 5
    for line in range(0,(len(content)-1),(nRows + 1)):
        time_stamp = content[line]
        img = []
        temp1 = content[(line+1):(line+(nRows + 1))]
        # for each line
        for i in range(0,nRows):
            temp2 = temp1[i].split()
            # convert each element into a pixel/temperature value
            temp3 = [float(x) for x in temp2]
            img.append(np.array(temp3))
        img = np.reshape(np.concatenate(img), [nRows, nCols])


        imgs.append(img)
        #imgs.append(img_4_5_1)
        time_stamps.append(time_stamp)

    # normalization
    max_t = float('-inf')
    min_t = float('inf')
    for i in range(len(imgs)):
        curr_max_t = np.max(imgs[i])
        curr_min_t = np.min((imgs[i])[np.nonzero(imgs[i])])
        if curr_max_t > max_t:
            max_t = curr_max_t
        if curr_min_t < min_t:
            min_t = curr_min_t
    
    mean_value = (max_t + min_t) / 2
    for i in range(len(imgs)):
        imgs[i][imgs[i] == 0] = mean_value         

    # rescale
    normalize = lambda x : ((x - min_t) / (max_t - min_t)) * 2 - 1
    vect_normalize = np.vectorize(normalize)
    imgs = vect_normalize(imgs)

    STEP_SIZE = 30
    inpts = []
    preds = []
    total_hours = np.int((len(content)-1)/(nRows + 1))
    for hour in range(0,(total_hours-STEP_SIZE)):
        print(hour)
        inpt = imgs[hour:hour+STEP_SIZE]
        pred = imgs[(STEP_SIZE+hour)]
        inpts.append(np.reshape(np.concatenate(inpt),[-1,4,5,1]))
        # preds.append(pred)
        preds.append(np.reshape(np.concatenate(pred),[-1,4,5,1]))

    # return_item_y = [0] * len(inpts)

    return np.array(inpts), np.array(preds)
    #np.reshape(np.array(preds),[-1, 99])
