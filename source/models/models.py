from keras import initializers, regularizers
from keras.layers import Dense, Dropout, CuDNNLSTM, Bidirectional
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
    adam = optimizers.Adam(lr=0.01)
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    ##########

    model = Sequential()
    model.add(CuDNNLSTM(64))
    model.add(Dense(20, activation='softmax', 
                    kernel_initializer='glorot_normal',
                    activity_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='softmax', 
                    kernel_initializer='glorot_normal',
                    activity_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model