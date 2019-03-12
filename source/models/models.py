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
    #model.add(Dropout(kwargs.get('dropout', 0.5)))

    model.add(Dense(dense_size, activation='sigmoid', kernel_initializer='normal'))
    #model.add(Dropout(kwargs.get('dropout', 0.5)))

    model.add(Dense(dense_size, activation='sigmoid'))

    model.add(Dense(kwargs.get('label_size', 2), activation='sigmoid'))
    
    #opt = optimizers.adam(lr=0.01)
    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
    
    #ada = optimizers.Adadelta(lr=0.01, rho=0.95, epsilon=None, decay=0.0)
    #model.compile(loss='mean_squared_error', optimizer=ada, metrics = ['accuracy'])
    return model
