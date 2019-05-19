from keras.models import load_model
import tensorflow as tf
import argparse
import json
import shutil
from os import path
import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error 
import utils
from data.data_stuff import data_testing
from contextlib import redirect_stdout

def load_model_to_estimate(modelName):
    parser = argparse.ArgumentParser(description='NA')
    parser.add_argument('-c', '--configure', default='drive/My Drive/ecg-final/configure/colab_classifier.json', help='JSON file')
    args = parser.parse_args()

    with open(args.configure) as f:
        config = json.load(f)

    model = load_model(modelName)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    X_test, Y_test = data_testing(config)
    y_pred = model.predict(X_test, batch_size=1024)

    y_pred = np.argmax(y_pred, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    f1 = classification_report(Y_test, y_pred)
    print('Accuracy: ',accuracy_score(Y_test, y_pred))
    print(f1)

    utils.save_dif_result(y_pred,Y_test,X_test,path.dirname(modelName))

load_model_to_estimate('drive/My Drive/data/mitbih_result/16-05-2019-09-44-26/cls-09-0.97.hdf5')
