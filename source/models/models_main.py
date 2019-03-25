import tensorflow as tf
from tensorflow import keras

def lstm_classifier(**kwargs):
    """
    A LSTM model for normal/abnormal classifier
    :param kwargs: include all parameters including: input_vector_size, dropout, ...
    :return: a model ready to training
    """
    print('code go here')

    model = keras.Sequential(
        # keras.layers.LSTM(input_vector_size),
        keras.layers.Dense(324,activation='relu',kernel_initializer='glorot_normal'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64,activation='relu',kernel_initializer='glorot_normal'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2,activation='sigmoid',kernel_initializer='glorot_normal')
    )
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
    )
    return model
