import numpy as np
import wfdb

physionet_path = '/home/trung/py/data/mitdb'
## Record path.
record_path = physionet_path + "/106"
record = wfdb.rdrecord(record_path, sampto=2500)
ann = wfdb.rdann(record_path, 'atr', sampto=2500)
wfdb.plot_wfdb(record=record, annotation=ann,
               title='Record 100 from MIT-BIH Arrhythmia Database',
               time_units='samples')


# nu_plus = 0
# for i in range(100,110,1):
#     x = wfdb.rdann(physionet_path+'/'+str(i),'atr',sampto=2500)
#     if x.symbol[0]=='+':
#         nu_plus += 1
#     else:
#         print(i)

# print(nu_plus)
# print(ann.__dict__)