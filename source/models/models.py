from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential


def lstm_classifier(**kwargs):
    """
    A LSTM model for normal/abnormal classifier
    :param kwargs: include all parameters including: input_vector_size, dropout, ...
    :return: a model ready to training
    """
    input_vector_size = kwargs.get('input_vector_size', 128)
    dense_size = kwargs.get('dense_size', 20)
    timesteps = None

    ##########

    model = Sequential()

    model.add(LSTM(int(input_vector_size), input_shape=(timesteps, int(input_vector_size)), return_sequences=False))
    model.add(Dropout(kwargs.get('dropout', 0.5)))

    model.add(Dense(dense_size, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=0)))
    model.add(Dropout(kwargs.get('dropout', 0.5)))

    model.add(Dense(dense_size, activation='softmax'))

    model.add(Dense(kwargs.get('label_size', 2), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
