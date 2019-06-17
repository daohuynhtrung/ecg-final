import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from scipy.signal import resample
import biosppy
import scipy.io
import os
import pandas
import csv
import scipy.signal as signal
import glob
import wfdb
from data.pca import reduceDemensionPCA

physionet_path_server = 'drive/My Drive/data/mitdb'
physionet_path_local = '/home/trung/py/data/mitdb'
kaggle = '/home/trung/py/data/kaggle_data/mitbih_test.csv'
local_file_mat = '/home/trung/py/data/mitbih'
local_file_csv = '/home/trung/py/data/mitbih_csv'
server_file_mat = '/data/mitbih'
server_file_csv = '/data/mitbih_csv'
#119(1543-444), 200(1743-856), 208(1586-994+373), 210(2423-32+194),223(2029-473+72+16+17), 228(1688-365)
normal_file = [100,101,103,105,106,112,113,114,115,116,117,121,122,123,201,202,205,209,213,215,219,220,222,234]
abnormal_file = [104,108,109,111,118,119,124,200,203,207,208,210,212,214,217,221,223,228,230,231,232]

def read_mat(file_name):
    mat = scipy.io.loadmat(file_name)
    
    return mat['val'][0]


def read_csv(file_name):
    data = pd.read_csv(file_name, header=None, engine='python')
    data = np.array(data)
    for i in range(20):
        pltfile_name = '/home/trung/py/data/kaggle_data/'+str(i)+'.png'
        plot_data(data=data[i],filename=pltfile_name)
    print('Done')
    

def plot_data(data, r_peak=list(), filename='plot.png', isSave=False):
    numeric = list(range(len(data)))
    plt.plot(numeric, data,'b-')
    temp = []
    for r in r_peak:
        temp.append(data[r])
    if r_peak!=list():
        plt.plot(r_peak, temp,'ro')
    
    if isSave==True:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
        plt.close()


def resamples():
    a = np.array([0,1])
    c = np.array([0,2,4,6,8])
    b = resample(a,5)
    print(b)



def r_detect(data, sampling_rate=360):
    peaks =  biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate=sampling_rate)[0]

    return peaks


def denoise(data):
    N  = 2    # Filter order
    Wn = 0.15  # Cutoff frequency
    B, A = signal.butter(N, Wn, 'low',output='ba')
    smooth_data = signal.filtfilt(B,A, data)

    Wn = 0.01
    B, A = signal.butter(N, Wn, 'high',output='ba')
    smooth_data = signal.filtfilt(B,A, smooth_data)

    return smooth_data


def data_preprocessing(filename, csv_filename):
    data = read_mat(filename)
    # denoised_data = data
    denoised_data = denoise(data)

    window = 10
    n=int(len(data)/window)
    i=0
    while i<window:
        temp_data = []
        for l in range(i*n,(i+1)*n):
            temp_data.append(denoised_data[l])

        r_peak = r_detect(temp_data)
        listR = [0]*len(temp_data)
        for r in r_peak:
            listR[r]='R'

        csv_data = []
        for j in range(len(temp_data)):
            csv_data.append([temp_data[j],listR[j]])
        
        name = os.path.splitext(os.path.basename(filename))[0]+ '_' +str(i)
        csvWritename = csv_filename+'/'+name+'.csv'
        csvFile = open(csvWritename, 'w')
        with csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(csv_data)
        print('Writed data to ',csvWritename)
        
        i+=1


def label_file():
    label = []
    for i in range(len(normal_file)):
        label.append([normal_file[i],0])
    for j in range(len(abnormal_file)):
        label.append([abnormal_file[j],1])
    return label


def transformData():
    dataFile_label = label_file()
    for dl in dataFile_label:
        label = '/Normal'
        if dl[1]==1:
            label='/Abnormal'
        data_preprocessing(local_file_mat+'/'+str(dl[0])+'/'+str(dl[0])+'m.mat',local_file_csv+label)

# transformData()

# data = read_csv('/home/trung/py/data/mitbih_csv/Normal/121m_0.csv')
# data = read_csv('/home/trung/py/data/Full_data_for_ML/Abnormal/100m1.csv')
# data = read_csv('/home/trung/py/data/kaggle_data/mitbih_test.csv')
 
def feature_extraction():
    no = '220'
    record_path = physionet_path_local + "/" + no
    record = wfdb.rdrecord(record_path,sampto=2500)
    # ann = wfdb.rdann(record_path, 'atr',sampto=2500)
    data = record.p_signal
    ml2_data = data[:,0]
    r_data = r_detect(ml2_data)
    
    # plot_data(ml2_data)
    plot_data(ml2_data,r_data)
    
    # plot_annotation(record, ann)
    # all_data = []
    # data = read_mat('/home/trung/py/data/mitbih/'+ no+'/'+no+'m.mat')[0:25000]
    # # data_x = scipy.signal.resample(data,int(500*len(data)/360))[0:2000]
    # # denoise_data = denoise(data)
    # # window_size = int(len(data)/2)
    # # total_r = []
    # # for i in range(0,len(data), window_size):
    # #     r_peak = r_detect(data[i:i+window_size])
    # #     total_r = np.concatenate((total_r,r_peak),axis=None)
    # # print(total_r)
    # # plot_data(data,ann.sample)
    # newdata = data.reshape((len(data),1))
    # numeric = list(range(1000))
    # newdata = reduceDemensionPCA(newdata,1)
    # # plot_data(data[0])
    # plt.subplot(2,1,1)
    # plt.plot(numeric, data[0:1000])
    # plt.subplot(2,1,2)
    # plt.plot(numeric, newdata[0][0:1000])
    # plt.show()

def plot_raw():
    no = '220'
    record_path = physionet_path_local + "/" + no
    record = wfdb.rdrecord(record_path,sampto=2500)
    # ann = wfdb.rdann(record_path, 'atr',sampto=2500)
    data = record.p_signal
    ml2_data = data[:,0]
    r_data = r_detect(ml2_data)
    
    plot_data(ml2_data)