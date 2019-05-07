import numpy as np
import wfdb
import random
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import glob
import csv
from source.data.plot_data import denoise, plot_data
from source.data.pca import reduceDemensionPCA, reduceDemensionICA
import matplotlib.pyplot as plt 
# from source.data.plot_data import r_detect

#not contain paced beat: 102, 104, 107, 217
mitbih_file = [100,108,113,117,122,201,207,212,222,231,101,105,109,114,118,123,202,208,213,219,223,232,106,111,115,119,124,203,209,214,220,228,233,103,112,116,121,200,205,210,215,221,230,234]
normal_files = [103,105,106,108,112,113,114,115,116,117,119,121,122,123,200,201,202,203,205,208,209,210,213,215,219,220,221,222,223,228,230,233,234]
abnormal_files = [111,124,207,212,214,231,232]

testing_set = [220, 233, 221]

physionet_path = '/home/trung/py/data/mitdb'
csv_file_path = '/home/trung/py/data/mitbih_wfdb'

# record_path = physionet_path + "/100"
# record = wfdb.rdrecord(record_path,sampto=2500)
# ann = wfdb.rdann(record_path, 'atr',sampto=2500)
# record_data = record.p_signal[:,0]
# record_data = denoise(record_data)
# plot_data(record_data,isSave=True,filename='plot3.png')


def plot_annotation(record,ann):
    wfdb.plot_wfdb(record=record, annotation=ann,
                   title='Record 100 from MIT-BIH Arrhythmia Database',
                   time_units='samples')


def write_csv(data,ann,filename, csv_filename):
    annotation = [0] * len(data)
    for i in range(len(ann.sample)):
        annotation[ann.sample[i]]=ann.symbol[i]
    csv_data = []
    for i in range(len(data)):
        csv_data.append([data[i],annotation[i]])

    name = os.path.splitext(os.path.basename(filename))[0]
    csvWritename = csv_filename+'/'+name+'.csv'
    csvFile = open(csvWritename, 'w')
    with csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csv_data)
    print('Writed data to ',csvWritename)


def data_processing():
    for file in mitbih_file:
        record_path = physionet_path+'/'+str(file)
        record = wfdb.rdrecord(record_path)
        ann = wfdb.rdann(record_path, 'atr')
        record_data = [ i[0] for i in record.p_signal ]
        write_csv(record_data,ann,record_path+'.dat',csv_file_path)


def rr_normalize_time(data,sampling_rate=360):
    x = np.arange(len(data))
    y = data
    xnew = np.linspace(0,len(data),sampling_rate*0.9)
    ynew = np.interp(xnew, x, y)
    return ynew


def rr_normalize_amplitude(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    data = scaler.fit_transform(data.reshape(-1,1)).flatten()
    return data


def rr_processing(file_name,k=0,offset=0):
    try:
        df = pd.read_csv(file_name, header=None, engine='python')

        values = df.loc[:][0].tolist()
        signal_type = df.loc[:][1].tolist()

        c = [i for i in range(len(signal_type)) if signal_type[i] != '0']
        # r_data = [values[c[i]:c[i + k]] for i in range(len(c) - k)]
        labels = []
        r_data = []
        for i in range(len(c)-k-3):
            data = np.array(values[ c[i] : (c[i+k]+offset)])
            data = rr_normalize_time(data,360*k)
            # data = rr_normalize_amplitude(data)
            r_data.append(data)
            if signal_type[c[i]]!='N' or signal_type[c[i+1]]!='N':
                labels.append(1.)
            else:
                labels.append(0.)
        return r_data,labels

    except Exception as e:
        print(e)
        print(file_name)
        return [], []


def data_prepare(files):
    
    all_data = []
    all_label = []
    for file in files:
        datas, labels = rr_processing(file_name=file,k=1)
        for data in datas:
            all_data.append(data)
        for label in labels:
            all_label.append(label)
    
    # return np.array(all_data), np.array(all_label)
    
    return np.array(all_data), to_categorical(np.array(all_label))


def data_pipeline_na_split(data_path):
    normal_file_split = [data_path+'/'+str(nfile)+'.csv' for nfile in normal_files]
    abnormal_file_split = [data_path+'/'+str(afile)+'.csv' for afile in abnormal_files]
    
    X_ntrain, X_ntest = train_test_split(normal_file_split,test_size=0.2,shuffle=True)
    X_atrain, X_atest = train_test_split(abnormal_file_split,test_size=0.2,shuffle=True)
    
    X_trainfiles = X_atrain + X_ntrain
    X_testfiles = X_atest + X_ntest

    X_train, y_train = data_prepare(X_trainfiles)
    X_train, y_train = oversampling(X_train, y_train)
    # X_test, y_test = data_prepare(X_testfiles)
    testing_list = [data_path+'/'+str(f)+'.csv' for f in testing_files]
    X_test, y_test = data_prepare(testing_list)

    return X_train, y_train, X_test, y_test

def oversampling(data, labels, multime=2):
    n_data = []
    a_data = []
    for i in range(len(data)):
        if np.array_equal(labels[i],[1,0]):
            n_data.append([data[i],labels[i]])
        else:
            a_data.append([data[i],labels[i]])
    n_data = random.sample(n_data,len(n_data))
    a_data = random.sample(a_data,len(a_data))
    
    if len(n_data) > len(a_data):
        n_data = n_data[0:len(a_data)]
    else:
        a_data = a_data[0:len(n_data)]
    # print(len(n_data))
    # print(len(a_data))
    data = a_data + n_data
    shuffle_data = random.sample(data,len(data))
    new_data = [ i[0] for i in shuffle_data ]
    new_label = [ i[1] for i in shuffle_data ]
    return np.array(new_data), np.array(new_label)
    
def oversampling_m2o(data, labels, multime=2):
    n_data = []
    a_data = []
    for i in range(len(data)):
        if labels[i]==0.0:
            n_data.append([data[i],labels[i]])
        else:
            a_data.append([data[i],labels[i]])
    a_data = a_data*multime
    n_data = random.sample(n_data,len(n_data))
    a_data = random.sample(a_data,len(a_data))
    
    if len(n_data) > len(a_data):
        n_data = n_data[0:len(a_data)]
    else:
        a_data = a_data[0:len(n_data)]
    # print(len(n_data))
    # print(len(a_data))
    data = a_data + n_data
    shuffle_data = random.sample(data,len(data))
    new_data = [ i[0] for i in shuffle_data ]
    new_label = [ i[1] for i in shuffle_data ]
    return np.array(new_data), np.array(new_label)

def data_pipeline(data_path):
    data_files = [i for i in glob.glob(data_path+'/*.csv')]

    X_trainfiles, X_testfiles = train_test_split(data_files,test_size=0.2,shuffle=True)

    X_train, y_train = data_prepare(X_trainfiles)
    X_train, y_train = oversampling(X_train, y_train)

    X_test, y_test = data_prepare(X_testfiles)
    # Xtrain = reduceDemensionPCA(X_train, 30)
    # Xtest = reduceDemensionPCA(X_test, 30)
    # testing_list = [data_path+'/'+str(f)+'.csv' for f in testing_files]
    # X_test, y_test = data_prepare(testing_list)
    return X_train, y_train, X_test, y_test

def data_pipeline_test_choosen(data_path):
    training_set = list(set(mitbih_file)-set(testing_set))
    training_files = [data_path+'/'+str(training_file)+'.csv' for training_file in training_set]
    testing_files = [data_path+'/'+str(testing_file)+'.csv' for testing_file in testing_set]
    
    X_train, y_train = data_prepare(training_files)
    # X_train, y_train = oversampling(X_train, y_train)

    X_test, y_test = data_prepare(testing_files)


    return X_train, y_train, X_test, y_test

def testing():
    X_train, y_train, X_test, y_test = data_pipeline('/home/trung/py/data/mitbih_wfdb_local') 
    # return
    new_Xtrain = reduceDemensionPCA(X_train, 30)

    for no in range(10):
        numeric_old = list(range(len(X_train[no])))

        numeric_new = list(range(len(new_Xtrain[no])))

        print(y_train[no])
        plt.subplot(2,1,1)
        plt.plot(numeric_old, X_train[no])
        plt.subplot(2,1,2)
        plt.plot(numeric_new, new_Xtrain[no])
        plt.show()

# testing()
# data_pipeline('/data/mitbih_wfdb')