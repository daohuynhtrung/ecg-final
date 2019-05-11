import calendar
import json
import importlib
import os
import time
from time import strftime, localtime
from datetime import datetime
import pytz
from time import gmtime, strftime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from os import path
import numpy as np

def load_func_by_name(func_str):
    """
    Load function by full name, e.g. gtopia_ml.data.DataHelper.rr_load_raw
    :param func_str:
    :return: fn, mod
    """
    mod_name, func_name = func_str.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    return func, mod


def make_dir_epoch_time(base_path):
    """
    make a new dir on base_path with epoch_time
    :param base_path:
    :return:
    """
    # t = calendar.timegm(time.gmtime())
    t = datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')).strftime("%d-%m-%Y-%H-%M-%S")
    new_path = "{}/{}".format(base_path, t)
    os.makedirs(new_path)
    return new_path

def plot_result_by_history(history, fileName):
    try:
        epoch = list(range(len(history['acc'])))
        plt.plot(epoch, history['acc'])
        plt.plot(epoch, history['val_acc'])
        plt.ylabel('%')
        plt.xlabel('epoch')
        plt.savefig(fileName + '/acc.png')
        plt.close()

        plt.plot(epoch, history['loss'])
        plt.plot(epoch, history['val_loss'])
        plt.xlabel('epoch')
        plt.savefig(fileName + '/loss.png')
        plt.close()

        return True
    except Exception as e:
        print(e)
        return False

def print_result_by_json(fileName):
    resultName = fileName + '/result.json'
    if resultName:
        with open(resultName, 'r') as f:
            datastore = json.load(f)
        epoch = list(range(len(datastore['acc'])))
        
        plt.plot(epoch, datastore['acc'])
        plt.plot(epoch, datastore['val_acc'])
        plt.ylabel('%')
        plt.xlabel('epoch')
        plt.savefig(fileName + '/acc.png')
        plt.close()

        plt.plot(epoch, datastore['loss'])
        plt.plot(epoch, datastore['val_loss'])
        plt.xlabel('epoch')
        plt.savefig(fileName + '/loss.png')
        plt.close()

def display_result(fileName):
    resultName = fileName + '/result.json'
    if resultName:
        with open(resultName, 'r') as f:
            datastore = json.load(f)
    print('loss\tacc\tval_loss\tval_acc')
    for i in range(len(datastore)):
        print(str(datastore['loss'][i])+'\t'+str(datastore['acc'][i])+'\t'+str(datastore['val_acc'][i])+'\t'+str(datastore['val_loss'][i]))
# display_result('/home/trung/py/data/mitbih_result/22-04-2019-08:45:28')

def display_f1_score(fileName):
    resultName = fileName + '/f1.json'
    if resultName:
        with open(resultName, 'r') as f:
            datastore = json.load(f)

        print(datastore)

def save_dif_result(y_pred, Y_test, X_test, checkpoint_path):
    n_dif = 0
    a_dif = 0
    total_n = 0
    total_a = 0
    for i in Y_test:
        if i==0:
            total_n+=1
        else:
            total_a+=1

    dif = [i for i in range(len(y_pred)) if y_pred[i]!=Y_test[i]]
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[1]))
    X_dif = [ np.ndarray.tolist(X_test[i]) for i in dif ]
    json.dump(X_dif,open(checkpoint_path+'/X_dif.json','w'))
    Y_dif = [ str(Y_test[i]) for i in dif ]
    json.dump(Y_dif,open(checkpoint_path+'/Y_dif.json','w'))

    for i in dif:
        if Y_test[i]==0:
            n_dif+=1
        else:
            a_dif+=1
    print('normal wrong: ', n_dif,'/',total_n)
    print('abnormal wrong: ', a_dif,'/',total_a)
