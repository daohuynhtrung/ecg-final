from keras import initializers
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras import optimizers
from keras.optimizers import SGD,adam,Adagrad,RMSprop,Adadelta
import tensorflow as tf


# local testing model
def lstm_classifier(**kwargs):
    """
    A LSTM model for normal/abnormal classifier
    :param kwargs: include all parameters including: input_vector_size, dropout, ...
    :return: a model ready to training
    """

    adam_opt = optimizers.Adam(lr=kwargs['lr'])
    ##########

    model = Sequential()

    model.add(LSTM(32))

    for i in kwargs['num_layers']:
        model.add(Dense(i, activation='softmax'))
    
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])    
    
    return model
