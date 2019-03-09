import calendar
import importlib
import os
import time
from time import strftime, localtime
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
    t = calendar.timegm(time.gmtime())
    new_path = "{}/{}".format(base_path, t)
    os.makedirs(new_path)
    return new_path

def plot_result(hist, config, i):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model plot')
    plt.ylabel('%')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss', 'acc', 'val_acc'], loc='upper right')
    outpath = config.get('plot_path', '/data')
    plotpath = outpath + "/"
    plt.savefig(path.join(plotpath, "plot_image_{0}.png".format(i)))
    plt.clf()
    print('Saved plot model')
    # plt.show()
    return 0

def mean_result(acc, loss, val_acc, val_loss, score, config, checkpoint_path):
    
    string = ''
    line = '====================================================================================\n'
    s_line = '____________________________________________________________________________________\n'
    string_data_time = strftime("%Y-%m-%d %H:%M:%S", localtime()) + '\n'

    fold_acc = []
    fold_loss = []
    fold_valacc = []
    fold_valloss = []

    string_all_fold = ''
    string_all_epoch = ''
    for fold_i in range(config['k']):

        #Mean fold result
        fold_nu = "Fold {0}: ".format(fold_i + 1) + '\n'
        mean_valacc_fold = "Val_acc: " + str(val_acc[fold_i][-1]) + '\n'
        mean_valloss_fold = "Val_acc: " + str(val_loss[fold_i][-1]) + '\n'
        acc_score = "Accuracy score: {:.2f}".format(score[fold_i]) + '\n'

        string_fold = fold_nu + mean_valacc_fold + mean_valloss_fold + acc_score + '\n'
        string_all_fold += string_fold

        fold_acc.append(acc[fold_i][-1])
        fold_loss.append(loss[fold_i][-1])
        fold_valacc.append(val_acc[fold_i][-1])
        fold_valloss.append(val_loss[fold_i][-1])

        #epoch result
        string_epoch = ''
        string_header_fold = 'FOLD ' + str(fold_i) + '\n' + s_line + 'epoch\tacc\tloss\tval_acc\tval_loss\n'
        for e in range(config['epochs']):
            string_epoch += str(e+1) + '\t' + '{:.3f}'.format(acc[fold_i][e]) +'\t'+ '{:.3f}'.format(loss[fold_i][e]) +'\t'+ '{:.3f}'.format(val_acc[fold_i][e]) +'\t'+ '{:.3f}'.format(val_loss[fold_i][e]) + '\n'
        string_all_epoch += string_header_fold + string_epoch + line
    
    mean_acc = np.mean(fold_acc)
    mean_loss = np.mean(fold_loss)
    mean_valacc = np.mean(fold_valacc)
    mean_valloss = np.mean(fold_valloss)
    string_mean_acc = 'Mean accuracy: ' + str(mean_acc) + '\n'
    string_mean_loss = 'Mean loss: ' + str(mean_loss) + '\n'
    string_mean_valacc = 'Mean val_acc: ' + str(mean_valacc) + '\n'
    string_mean_valloss = 'Mean val_loss: ' + str(mean_valloss) + '\n'
    string_mean = string_mean_acc + string_mean_loss + string_mean_valacc + string_mean_valloss
    
    print(string_mean + line + string_all_fold)
    string_result = string_data_time + line + string_mean + line + string_all_fold + line + string_all_epoch

    f = open(checkpoint_path+'/result.txt','w')
    f.write(string_result)

    return 0