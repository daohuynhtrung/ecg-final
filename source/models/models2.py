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
    timesteps = None
    xav_init = tf.contrib.layers.xavier_initializer()
    ##########

    model = Sequential()

    model.add(LSTM(int(input_vector_size), input_shape=(timesteps, int(input_vector_size)), return_sequences=False))

    model.add(Dense(dense_size, activation='softmax', kernel_initializer='glorot_normal'))

    model.add(Dense(dense_size, activation='softmax', kernel_initializer='glorot_normal'))

    model.add(Dense(output, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
