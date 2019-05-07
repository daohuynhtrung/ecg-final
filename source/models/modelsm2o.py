from keras import initializers, regularizers
from keras.layers import Dense, Dropout, LSTM, Bidirectional, CuDNNLSTM
from keras.models import Sequential
from keras import optimizers
from keras.optimizers import SGD,adam,Adagrad,RMSprop,Adadelta
import tensorflow as tf

def lstm_classifier(**kwargs):
    """
    A LSTM model for normal/abnormal classifier
    :param kwargs: include all parameters including: input_vector_size, dropout, ...
    :return: a model ready to training
    """

    model = Sequential()
    model.add(CuDNNLSTM(64))
    # model.add(Dropout(0.2))
    model.add(Dense(20, 
                    activation='sigmoid', 
                    kernel_initializer='glorot_normal',
                    activity_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(20, 
                    activation='sigmoid', 
                    kernel_initializer='glorot_normal',
                    activity_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

