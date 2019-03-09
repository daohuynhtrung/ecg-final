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
    timesteps = None
    xav_init = tf.contrib.layers.xavier_initializer()
    ##########
    
    model = Sequential()

    model.add(LSTM(int(input_vector_size), input_shape=(timesteps, int(input_vector_size)), return_sequences=False))
    
    model.add(Dense(dense_size, activation='tanh', kernel_initializer=xav_init))

    model.add(Dense(dense_size, activation='sigmoid'))

    model.add(Dense(kwargs.get('label_size', 2), activation='sigmoid'))
    
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    
    return model
