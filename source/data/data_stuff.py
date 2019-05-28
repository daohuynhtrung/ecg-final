import numpy as np
import importlib
import wfdb
import random
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import glob
import csv
# from data.plot_data import denoise, plot_data
from data.pca import reduceDemensionPCA, reduceDemensionICA, PCA_reduce
import matplotlib.pyplot as plt 
import json

#not contain paced beat: 102, 104, 107, 217
mitbih_file = [100,108,113,117,122,207,212,222,231,101,105,109,114,118,123,202,208,213,219,223,232,106,111,115,119,124,203,209,214,220,228,233,103,112,116,121,200,205,210,215,221,230,234]

normal_files  = [100,101,103,105,106,112,113,114,115,116,117,121,122,123,202,205,209,213,215,219,220,222,234]
abnormal_files = [108,109,111,118,119,124,200,203,207,208,210,212,214,221,223,228,230,231,232,233]

# normal_files = [103,105,106,108,112,113,114,115,116,117,119,121,122,123,200,202,203,205,208,209,210,213,215,219,220,221,222,223,228,230,233,234]
# abnormal_files = [111,124,207,212,214,231,232]

# main testing set
testing_set0 = [220, 233, 221]
testing_set1 = [220, 233, 221, 103, 230, 116, 205]

# test normal/abnormal
normal_test_files1 = [100,101,103,106]
abnormal_test_files1 = [108,207,203,210]


physionet_path = '/home/trung/py/data/mitdb'
csv_file_path = '/home/trung/py/data/mitbih_wfdb'


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

def get_record_note(file_name):
    record_name = os.path.dirname(file_name) 
    record = os.path.splitext(os.path.basename(file_name))[0]
    with open(record_name+'/mitbih_record_note.json','r') as f:
        datastore = json.load(f)

    record_age = datastore[record][1]
    record_gender = datastore[record][0]
   
    return record_age, record_gender

def load_function(func_str):
    mod_name, func_name = func_str.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    return func

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


def rr_processing(file_name, timestep=0, k=0,offset=0):
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
            rr_length = len(data)
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


def rr_processing_one_for_all_timestep(file_name, timestep, k=0, offset=0):
    try:

        df = pd.read_csv(file_name, header=None, engine='python')
        record_age, record_gender = get_record_note(file_name)
   
        values = df.loc[:][0].tolist()
        signal_type = df.loc[:][1].tolist()

        c = [i for i in range(len(signal_type)) if signal_type[i] != '0']
        # r_data = [values[c[i]:c[i + k]] for i in range(len(c) - k)]
        labels = []
        r_data = []
        for i in range(len(c)-k-3):
            data = np.array(values[ c[i] : (c[i+k]+offset)])
            rr_length = len(data)
            data = rr_normalize_time(data,360*k)
            r_data.append(data)
            if signal_type[c[i]]!='N' or signal_type[c[i+1]]!='N':
                labels.append(1.)
            else:
                labels.append(0.)
        # old_data = r_data
        r_data = reduceDemensionPCA(r_data, 0.9)
        # plot_some(old_data,new_data)

        # concate record info
        concatenated_data = []
        for data in r_data:
            # scale and concate record info
            concatenated_data.append(np.concatenate((data,[rr_length/10, record_age/10, record_gender*10])))
        r_data = concatenated_data

        new_label = []
        new_data = []
        for i in range(len(r_data)-timestep+1):
            sample = r_data[i:(i+timestep)]
            new_data.append(sample)
            if(labels[i:(i+timestep)].count(1.)>2):
                new_label.append(1.)
            else:
                new_label.append(0.)

        return new_data,new_label

    except Exception as e:
        print(e)
        print('loi o file: ',file_name)
        return [], []

def rr_processing_concate_info(file_name, timestep=0,k=0,offset=0):
    try:

        df = pd.read_csv(file_name, header=None, engine='python')
        record_age, record_gender = get_record_note(file_name)
   
        values = df.loc[:][0].tolist()
        signal_type = df.loc[:][1].tolist()

        c = [i for i in range(len(signal_type)) if signal_type[i] != '0']
        # r_data = [values[c[i]:c[i + k]] for i in range(len(c) - k)]
        labels = []
        r_data = []
        for i in range(len(c)-k-3):
            data = np.array(values[ c[i] : (c[i+k]+offset)])
            rr_length = len(data)
            data = rr_normalize_time(data,360*k)
            r_data.append(data)
            if signal_type[c[i]]!='N' or signal_type[c[i+1]]!='N':
                labels.append(1.)
            else:
                labels.append(0.)

        # concate record info
        final_data = []
        for data in r_data:
            # scale and concate record info
            final_data.append(np.concatenate((data,[rr_length/10, record_age/10, record_gender*10])))

        return final_data,labels

    except Exception as e:
        print(e)
        print('loi o file: ',file_name)
        return [], []


def rr_processing_na(file_name, label, timestep, k=1, offset=0):
    try:
        record_age, record_gender = get_record_note(file_name)

        df = pd.read_csv(file_name, header=None, engine='python')

        values = df.loc[:][0].tolist()
        signal_type = df.loc[:][1].tolist()

        c = [i for i in range(len(signal_type)) if signal_type[i] != '0']
        # r_data = [values[c[i]:c[i + k]] for i in range(len(c) - k)]
        r_data = []
        for i in range(len(c)-k-3):
            data = np.array(values[ c[i] : (c[i+k]+offset)])
            rr_length = len(data)
            data = rr_normalize_time(data,360*k)
            r_data.append(data)
        
        # concate record info
        concatenated_data = []
        for data in r_data:
            # scale and concate record info
            concatenated_data.append(np.concatenate((data,[rr_length/10, record_age/10, record_gender*10])))
        r_data = concatenated_data

        # new_data = []
        # for i in range(len(r_data)-timestep+1):
        #     sample = r_data[i:(i+timestep)]
        #     new_data.append(sample)

        # labels = [label]*len(new_data)

        # return new_data,labels
        return r_data, [label]*len(r_data)

    except Exception as e:
        print(e)
        print(file_name)
        return [], []


def rr_processing_multi_classes(file_name,k=0,offset=0):
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
            if signal_type[c[i]]=='L' or signal_type[c[i+1]]=='L':
                labels.append(1.)
            elif signal_type[c[i]]=='R' or signal_type[c[i+1]]=='R':
                labels.append(2.)
            elif signal_type[c[i]]=='A' or signal_type[c[i+1]]=='A':
                labels.append(3.)
            elif signal_type[c[i]]=='V' or signal_type[c[i+1]]=='V':
                labels.append(4.)
            elif signal_type[c[i]]=='N':
                labels.append(0.)
            else:
                labels.append(5.)

        return r_data,labels

    except Exception as e:
        print(e)
        print(file_name)
        return [], []


def data_prepare(files, config):
    rr_processing_func = load_function(config['rr_processing_function'])
    timestep = config['timestep']

    all_data = []
    all_label = []
    for file in files:
        datas, labels = rr_processing_func(file_name=file, timestep=timestep, k=1)
        for data in datas:
            all_data.append(data)
        for label in labels:
            all_label.append(label)
    
    # return np.array(all_data), np.array(all_label)
    all_data = np.array(all_data)
    # all_data = all_data.reshape((all_data.shape[0],1,all_data.shape[1],1))
    
    return all_data, to_categorical(np.array(all_label))


def data_prepare_na_timestep(files, labels, config):
    timestep = config['timestep']
    all_data = []
    all_label = []
    for i in range(len(files)):
        new_data, new_label = rr_processing_na(files[i], labels[i], timestep)
        
        for data in new_data:
            all_data.append(data)
        for label in new_label:
            all_label.append(label)

    return np.array(all_data), to_categorical(np.array(all_label))

def oversampling(data, labels, multime=2):
    n_data = []
    a_data = []
    for i in range(len(data)):
        if np.array_equal(labels[i],[1,0]):
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
    data = a_data + n_data
    shuffle_data = random.sample(data,len(data))
    new_data = [ i[0] for i in shuffle_data ]
    new_label = [ i[1] for i in shuffle_data ]
    return np.array(new_data), np.array(new_label)

def oversampling_multi_classes(data, labels):
    N_data = []
    L_data = []
    R_data = []
    A_data = []
    V_data = []
    O_data = []

    for i in range(len(data)):
        if np.array_equal(labels[i],[1,0,0,0,0,0]):
            N_data.append([data[i],labels[i]])
        elif np.array_equal(labels[i],[0,1,0,0,0,0]):
            L_data.append([data[i],labels[i]])
        elif np.array_equal(labels[i],[0,0,1,0,0,0]):
            R_data.append([data[i],labels[i]])
        elif np.array_equal(labels[i],[0,0,0,1,0,0]):
            A_data.append([data[i],labels[i]])
        elif np.array_equal(labels[i],[0,0,0,0,1,0]):
            V_data.append([data[i],labels[i]])
        else:
            O_data.append([data[i],labels[i]])

    # UpSampling
    L_data = L_data*2
    R_data = R_data*2
    A_data = A_data*3
    V_data = V_data
    O_data = O_data

    # Shuffle
    N_data = random.sample(N_data,len(N_data))
    L_data = random.sample(L_data,len(L_data))
    R_data = random.sample(R_data,len(R_data))
    A_data = random.sample(A_data,len(A_data))
    V_data = random.sample(V_data,len(V_data))
    O_data = random.sample(O_data,len(O_data))

    #Down Sampling
    N_data = N_data[0:40000]

    data = N_data + L_data + R_data + A_data + V_data
    shuffle_data = random.sample(data,len(data))
    new_data = [ i[0] for i in shuffle_data ]
    new_label = [ i[1] for i in shuffle_data ]
    return np.array(new_data), np.array(new_label)

def data_pipeline(config):
    data_prepare_func = load_function(config['data_prepare_function'])

    data_files = [i for i in glob.glob(config['data_path']+'/*.csv')]

    X_trainfiles, X_testfiles = train_test_split(data_files,test_size=0.2,shuffle=True)

    X_train, y_train = data_prepare_func(X_trainfiles, config)
    X_test, y_test = data_prepare_func(X_testfiles, config)

    # Using to oversampling data
    # X_train, y_train = oversampling(X_train, y_train)

    # Using to denoise
    # Xtrain = reduceDemensionPCA(X_train, 30)
    # Xtest = reduceDemensionPCA(X_test, 30)
    return X_train, y_train, X_test, y_test

def data_pipeline_na(config):
    normal_train = list(set(normal_files)-set(normal_test_files1))
    abnormal_train = list(set(abnormal_files)-set(abnormal_test_files1))
    normal_test = normal_test_files1
    abnormal_test = abnormal_test_files1

    X_ntrain = [config['data_path']+'/'+str(nfile)+'.csv' for nfile in normal_train]
    X_atrain = [config['data_path']+'/'+str(afile)+'.csv' for afile in abnormal_train]
    X_ntest = [config['data_path']+'/'+str(nfile)+'.csv' for nfile in normal_test]
    X_atest = [config['data_path']+'/'+str(nfile)+'.csv' for nfile in abnormal_test]
    
    y_ntrain = [0.]*len(X_ntrain)
    y_atrain = [1.]*len(X_atrain)
    y_ntest = [0.]*len(X_ntest)
    y_atest = [1.]*len(X_atest)

    X_trainfiles = X_atrain + X_ntrain
    X_testfiles = X_atest + X_ntest
    y_trainfiles = y_ntrain + y_atrain
    y_testfiles = y_ntest + y_atest

    X_train, y_train = data_prepare_na_timestep(X_trainfiles, y_trainfiles, config)
    X_test, y_test = data_prepare_na_timestep(X_testfiles, y_testfiles, config)

    # data_prepare_func = load_function(config['data_prepare_function'])
    # X_train, y_train = data_prepare_func(X_trainfiles, y_trainfiles, config)
    # X_test, y_test = data_prepare_func(X_testfiles, y_testfiles, config)

    # X_train = X_train.reshape((X_train.shape[0],1, X_train.shape[1]))
    # X_test = X_test.reshape((X_test.shape[0],1, X_test.shape[1]))

    print(X_train.shape)
    print(X_test.shape)

    return X_train, y_train, X_test, y_test


def data_pipeline_test_choosen(config):
    testing_set = testing_set1
    training_set = list(set(mitbih_file)-set(testing_set))
    training_files = [config['data_path']+'/'+str(training_file)+'.csv' for training_file in training_set]
    testing_files = [config['data_path']+'/'+str(testing_file)+'.csv' for testing_file in testing_set]
    
    X_train, y_train = data_prepare(training_files, config)
    X_test, y_test = data_prepare(testing_files, config)

    # X_train, y_train = oversampling(X_train, y_train)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    return X_train, y_train, X_test, y_test

def data_testing(config):
    data_prepare_func = load_function(config['data_prepare_function'])
    testing_set = testing_set1
    testing_files = [config['data_path']+'/'+str(testing_file)+'.csv' for testing_file in testing_set]
    X_test, y_test = data_prepare_func(testing_files, config)
    return X_test, y_test


def plot_some(old_data, new_data):
    for i in range(10,20,1):
        raw_data = old_data[i]
        denoise_data = new_data[i]
        numeric = list(range(len(raw_data)))
        plt.subplot(211)
        plt.plot(numeric, raw_data)
        plt.subplot(212)
        plt.plot(numeric, denoise_data)
        plt.show()
        plt.close()

# def testing():
#     all = normal_class + abnormal_class
#     a = list(set(all)-set(mitbih_file))
#     b = list(set(mitbih_file)-set(all))
#     print(len(all))

# dataname = '/home/trung/py/data/mitbih_wfdb/213.csv'
# rr_processing_one_for_all_timestep(file_name=dataname, timestep=20, k=1)