from keras import initializers
from keras.layers import Dense, Dropout, LSTM
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
    input_vector_size = kwargs.get('input_vector_size', 128)
    dense_size = kwargs.get('dense_size', 20)
    output = kwargs.get('label_size', 2)
    timesteps = 1
    xav_init = tf.contrib.layers.xavier_initializer()
    ##########

    model = Sequential()
    # model.add(LSTM(input_vector_size, input_shape=(timesteps, input_vector_size)))
    model.add(LSTM(32 ,dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(20, activation='softmax', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='softmax', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
