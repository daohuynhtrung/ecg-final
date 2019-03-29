import numpy as np
import wfdb
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import glob
import csv

#not contain paced beat: 102, 104, 107, 217
mitbih_file = [100,108,113,117,122,201,207,212,222,231,101,105,109,114,118,123,202,208,213,219,223,232,106,111,115,119,124,203,209,214,220,228,233,103,112,116,121,200,205,210,215,221,230,234]
physionet_path = '/home/trung/py/data/mitdb'
csv_file_path = '/home/trung/py/data/mitbih_wfdb'

# record_path = physionet_path + "/100"
# record = wfdb.rdrecord(record_path,sampto=2500)
# ann = wfdb.rdann(record_path, 'atr',sampto=650000)
# print(ann.__dict__)


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
            data = np.array(values[ c[i] : (c[i+k]+offset) ])
        
            normalized_ampl_data = rr_normalize_amplitude(data)
            r_data.append(normalized_ampl_data)
            if signal_type[c[i]]!='N' or signal_type[c[i+1]]!='N':
                labels.append(1.)
            else:
                labels.append(0.)
        return r_data,labels

    except Exception as e:
        print(e)
        print(file_name)
        return [], []

def rr_normalize_amplitude(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    data = scaler.fit_transform(data.reshape(-1,1)).flatten()
    return data

def rr_normalize_time(data, size=324, x_size=360):
    time = len(data) / x_size
    x = np.arange(len(data)) / x_size
    xnew = np.linspace(0, time, size)
    y = data
    ynew = np.interp(xnew, x, y)
    return ynew

def data_prepare(files):
    
    all_data = []
    all_label = []
    for file in files:
        datas, labels = rr_processing(file_name=file,offset=450)
        for data in datas:
            all_data.append(data)
        for label in labels:
            all_label.append(label)
    
    return np.array(all_data), to_categorical(np.array(all_label))


def data_pipeline(data_path,k=1,load_data_fn=0):
    data_files = [i for i in glob.glob(data_path+'/*.csv')]
    X_trainfiles, X_testfiles = train_test_split(data_files,test_size=0.2,shuffle=True)
    
    X_train, y_train = data_prepare(X_trainfiles)
    X_test, y_test = data_prepare(X_testfiles)
    
    return X_train, y_train, X_test, y_test
