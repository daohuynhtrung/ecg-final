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

kaggle = '/home/trung/py/data/kaggle_data/mitbih_test.csv'
file_mat = '/home/trung/py/data/mitbih/'

def read_mat(file_name):
    mat = scipy.io.loadmat(file_name)
    
    return mat['val'][0]


def read_csv(file_name):
    data = pd.read_csv(file_name, header=None, engine='python')
    data = np.array(data)

    return data

    
def plot_data(data, r_peak, filename):
    temp = []
    for r in r_peak:
        temp.append(data[r])
    numeric = list(range(len(data)))
    plt.plot(numeric, data,'b-',r_peak, temp,'ro')
    # plt.show()
    plt.savefig(filename)


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


def data_preprocessing(filename):
    data = read_mat(filename)
    r_denoise = r_detect(denoised_data)
    # print(r_peak.shape)
    return

    listR = [0]*len(denoised_data)
    for r in r_peak:
        listR[r]='R'

    csv_data = []
    for i in range(len(denoised_data)):
        csv_data.append([denoised_data[i],listR[i]])

    return csv_data


def write_csv(file_name, data):
    csvFile = open(file_name, 'w')
    with csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)
    print('Writed data to ',file_name)

# data = data_preprocessing(file_mat+'100/100m.mat')

data = read_mat(file_mat+'100/100m.mat')
sumr = 0
window = 100
n=int(len(data)/window)
i=0
while i<window:
    mind = i*n
    maxd = (i+1)*n
    temp_data = []
    for l in range(mind,maxd):
        temp_data.append(data[l])
    temp_data = denoise(temp_data)
    r_peak = r_detect(temp_data)
    plot_data(temp_data,r_peak, '/home/trung/py/data/mitbih_csv/'+str(i)+'.png')
    sumr += len(r_peak)
    i+=1
    
print(sumr)
# plot_data(data, r_peak)