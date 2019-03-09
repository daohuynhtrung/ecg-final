import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


def read_data():
    myarray = np.fromfile('/home/trung/py/data/MITBDH/dat/100 .dat', dtype=float)
    plt.plot(myarray[0:1000])
    plt.ylabel('sample')
    plt.show()

    # print(myarray[0:500])

def read_csv():
    data = pd.read_csv('/home/trung/py/data/kaggle_data/mitbih_test.csv')
    print(data[0])
    # plt.plot(data.head(1))
    # plt.ylabel('fwfa')
    # plt.show()

read_csv()
