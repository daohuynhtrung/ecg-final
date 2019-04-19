import json
import random
from multiprocessing.pool import Pool

import numpy as np
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.utils import to_categorical

def single_rr_load(file_name, label):
    """
    Load file, get all R-R and return list of R-R, label will be repeated to get same size of R-R.
    File will be array of R-R, must be normalize to same size for all R-R
    :param file_name:
    :param label:
    :return:
    """
    x = np.load(file_name)
    return x, [label] * (len(x))


def single_rr_load_one(one):
    """
    single_rr_load with tuple param
    :param one: is file_name, label
    :return:
    """
    single_rr_load(one[0], one[1])


def load_single_rr_fn_o(o):
    return single_rr_load(o[0], o[1])


def load_n_samples_unnormalize(file_name, label, samples=1200):
    """

    """
    try:
        df = pd.read_csv(file_name, header=None, engine='python')

        values = df.loc[:][0].tolist()
        signal_type = df.loc[:][1].tolist()

        c = [i for i in range(len(signal_type)) if signal_type[i] == 'R']
        r_data = []
        for i in range(len(c)):
            if (c[i] + samples) > len(values):
                break
            r_data.append(values[c[i]:(c[i] + samples)])

        # data = rr_normalize_amplitude(r_data)
        data = list(map(rr_normalize_amplitude(samples),r_data))

        return data, [0]*20
    except Exception as e:
        print(e)
        print(file_name)
        return [], []


def load_1200_samples_unnormalize(o):
    file_name, label = o
    return load_n_samples_unnormalize(file_name, label)

def load_1600_samples_unnormalize(o):
    file_name, label = o
    return load_n_samples_unnormalize(file_name, label, 1600)

def load_600_samples_unnormalize(o):
    file_name, label = o
    return load_n_samples_unnormalize(file_name, label,600)

def rr_normalize_amplitude(size=1200):

    def fn(r_data):
        r_data = np.array(r_data)
        amp = max(r_data)
        min_val = min(r_data)
        if amp < abs(min_val):
            amp = abs(min_val)
        
        return r_data / amp

    return fn

def rr_load_raw(file_name, label, k=1):
    """
    similar as `single_rr_load but from raw data instead npy with normal...
    cut and get single R-R from raw data
    :param file_name: csv file, 2 columns, first is raw data, second is PRQS
    :param label:
    :param k: number of R-R, default is single R-R
    :return: [None, data], [None], data may be n-dimension vector
    """
    try:
        # read csv
        df = pd.read_csv(file_name, header=None, engine='python')

        # scan and get R-R
        values = df.loc[:][0].tolist()
        signal_type = df.loc[:][1].tolist()

        c = [i for i in range(len(signal_type)) if signal_type[i] == 'R']
        r_data = [values[c[i]:c[i + k]] for i in range(len(c) - k)]

        # normalize_rr, after that, data will be [None, n-dimension] and same size
        data = list(map(rr_normalize_fn(324 * k, 360 * k), r_data))

        # collect results
        return data, [label] * len(data)
    except Exception as e:
        print(e)
        print(file_name)
        return [], []


def rr_load_raw_f1(o):
    """
    Wrap for map/pool.map just input one param
    :param o:
    :return:
    """
    file_name, label = o
    return rr_load_raw(file_name=file_name, label=label)


def rr_load_raw_f2(o):
    file_name, label = o
    return rr_load_raw(file_name=file_name, label=label, k=2)


def rr_load_raw_f3(o):
    file_name, label = o
    return rr_load_raw(file_name=file_name, label=label, k=3)


def rr_load_raw_f5(o):
    file_name, label = o
    return rr_load_raw(file_name=file_name, label=label, k=5)


def rr_normalize_fn(size=450, x_size=500):
    """
    return a function to call with data (a RR)
    :param size: size to normalize to
    :param x_size: ?
    :return:
    """
    
    def fn(rr_data):
        """
        normalize data for R-R
        :param rr_data: one R-R
        :return: normalize to `size`
        """
        time = len(rr_data) / x_size
        x = np.arange(len(rr_data)) / x_size
	
        xnew = np.linspace(0, time, size)

        y = rr_data
        ynew = np.interp(xnew, x, y)

        # if abs(max(ynew)) > abs(min(ynew)):
        #     return ynew / abs(max(ynew))
        return (ynew - min(ynew)) / max(ynew)
    return fn


def k_fold(files, labels, k=3, load_data_fn=single_rr_load):
    """

    :param k:
    :param files: name list
    :param labels: label list, same size with files
    :param load_data_fn: a function received file_name/data/raw_data and label, return (list of data, list of label) same size
    :return: x_train, y_train, x_test, y_test
    """
    kf = KFold(n_splits=k, shuffle=True)

    def f_load(file_names, labels):
        x_data, y_data = [], []
        for x_file, y in zip(file_names, labels):
            x_data_l, y_data_l = load_data_fn(x_file, y)
            x_data.extend(x_data_l)
            y_data.extend(y_data_l)

        return np.array(x_data), to_categorical(np.array(y_data))

    def f_load_parallel(file_names, labels, kthread=16):
        arr = zip(file_names, labels)

        x_data, y_data = [], []
        with Pool(kthread) as p:
            r = p.map(load_data_fn, arr)
            for x_data_l, y_data_l in r:
                x_data.extend(x_data_l)
                y_data.extend(y_data_l)

        return np.array(x_data), to_categorical(np.array(y_data))

    for train_index, test_index in kf.split(files):
        x_train_files, x_test_files = files[train_index], files[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # mapping original file to current file
        #x_train_files, y_train = mapping_file(x_train_files, y_train)
        #x_test_files, y_test = mapping_file(x_test_files, y_test)

        x_train_data, y_train_data = f_load_parallel(x_train_files, y_train)
        x_test_data, y_test_data = f_load_parallel(x_test_files, y_test)

        print('y test d:', y_test_data.shape)
        print('y train d:', y_train_data.shape)
        yield x_train_data, y_train_data, x_test_data, y_test_data


def data_prepare(files, labels, test_files, test_labels, load_data_fn=single_rr_load):
    """

    :param files: name list
    :param labels: label list, same size with files
    :param load_data_fn: a function received file_name/data/raw_data and label, return (list of data, list of label) same size
    :return: x_train, y_train, x_test, y_test
    """

    def f_load_parallel(file_names, labels, kthread=16):
        arr = zip(file_names, labels)

        x_data, y_data = [], []
        with Pool(kthread) as p:
            r = p.map(load_data_fn, arr)
            for x_data_l, y_data_l in r:
                x_data.extend(x_data_l)
                y_data.extend(y_data_l)
        
        # return np.array(x_data), np.array(y_data)
        return np.array(x_data), to_categorical(np.array(y_data))

    x_train_files, x_test_files, y_train, y_test = train_test_split(files, labels, test_size=0.1)
    # x_test_files = np.concatenate((x_train_files,test_files))
    # y_test = np.concatenate((y_test,test_labels))
    x_test_files = test_files
    y_test = test_labels

    x_train_data, y_train_data = f_load_parallel(x_train_files, y_train)
    x_test_data, y_test_data = f_load_parallel(x_test_files, y_test)

    return x_train_data, y_train_data, x_test_data, y_test_data


def mapping_file(files, labels):
    """
    mapping from original file to current file
        :param files:
        :param labels:
    """
    current_files = []
    current_labels = []
    for file_name in files:
        dir_name = os.path.dirname(file_name)
        file_index = files.tolist().index(file_name)
        splited_files = []
        splited_labels = []
        for f in os.listdir(dir_name):
            f_name = dir_name + '/' + f
            if revert_to_original_filename(f_name) == file_name:
                splited_files.append(f_name)
                splited_labels.append(labels[file_index])
        current_files.append(splited_files)
        current_labels.append(splited_labels)

    current_files = np.hstack(np.asarray(current_files))
    current_labels = np.hstack(np.asarray(current_labels))

    return current_files, current_labels


def get_original_name(s):
    """
    get original file name
        :param s:
    """
    while 1:
        if s[-1].isdigit():
            s = s[0:-1]
        else:
            break
    return s


def revert_to_original_filename(filename):
    """
    revert a original file name from the current file name
        :param filename:
    """
    base_name = os.path.splitext(os.path.basename(filename))[0]
    extend_name = os.path.splitext(os.path.basename(filename))[1]
    original_base_name = get_original_name(base_name)
    return os.path.dirname(filename) + '/' + original_base_name + extend_name


def k_fold_single_rr_data_pipeline(data_path, k=3, load_data_fn=single_rr_load):
    """

    :param k: number of fold
    :param data_path:
    :return:
    """
    normal_files = ['{}/Normal/{}'.format(data_path, o) for o in os.listdir('{}/Normal'.format(data_path))]
    abnormal_files = ['{}/Abnormal/{}'.format(data_path, o) for o in os.listdir('{}/Abnormal'.format(data_path))]

    # normal_original_files = list(set([revert_to_original_filename(o) for o in normal_files]))
    # abnormal_original_files = list(set([revert_to_original_filename(o) for o in abnormal_files]))

    normal_original_files = normal_files
    abnormal_original_files = abnormal_files

    files = normal_original_files + abnormal_original_files
    labels = [0] * len(normal_original_files) + [1] * len(abnormal_original_files)
    files, labels = np.array(files), np.array(labels)

    for all_data in k_fold(files=files, labels=labels, k=k, load_data_fn=load_data_fn):
        yield all_data


def data_pipeline(data_path, k=3, load_data_fn=single_rr_load):
    normal_files = ['{}/Normal/{}'.format(data_path, o) for o in os.listdir('{}/Normal'.format(data_path))]
    abnormal_files = ['{}/Abnormal/{}'.format(data_path, o) for o in os.listdir('{}/Abnormal'.format(data_path))]
    normal_test_files = ['{}/Test/Normal/{}'.format(data_path, o) for o in os.listdir('{}/Test/Normal'.format(data_path))]
    abnormal_test_files = ['{}/Test/Abnormal/{}'.format(data_path, o) for o in os.listdir('{}/Test/Abnormal'.format(data_path))]
    # normal_original_files = list(set([revert_to_original_filename(o) for o in normal_files]))
    # abnormal_original_files = list(set([revert_to_original_filename(o) for o in abnormal_files]))

    normal_original_files = normal_files
    abnormal_original_files = abnormal_files

    files = normal_original_files + abnormal_original_files
    labels = [0] * len(normal_original_files) + [1] * len(abnormal_original_files)
    files, labels = np.array(files), np.array(labels)

    test_files = normal_test_files + abnormal_test_files
    test_labels = [0] * len(normal_test_files) + [1] * len(abnormal_test_files)
    test_files, test_labels = np.array(test_files), np.array(test_labels)

    return data_prepare(files=files, labels=labels, test_files=test_files, test_labels=test_labels, load_data_fn=load_data_fn)