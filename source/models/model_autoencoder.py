import json
import os
import sys
import argparse
import numpy as np
import random
from numpy import array
import theano
import tensorflow as tf
import keras.callbacks

def model(input_size, nb_epoch, batch_size, X_train, X_test):
    learning_rate = 0.001
    xav_init = tf.contrib.layers.xavier_initializer()
    l2_regularizer = tf.contrib.layers.l2_regularizer(0.01)
    optimize_adam = tf.train.AdamOptimizer(learning_rate)

    input_data = tf.keras.layers.Input(shape=(input_size,))
    encoded = tf.keras.layers.Dense(256, 
                                    activation='tanh', 
                                    kernel_initializer=xav_init,
                                    kernel_regularizer=l2_regularizer)(input_data)
    encoded = tf.keras.layers.Dense(128, 
                                    activation='tanh', 
                                    kernel_initializer=xav_init,
                                    kernel_regularizer=l2_regularizer)(encoded)

    decoded = tf.keras.layers.Dense(256, 
                                    activation='tanh', 
                                    kernel_initializer=xav_init,
                                    kernel_regularizer=l2_regularizer)(encoded)
    decoded = tf.keras.layers.Dense(input_size, 
                                    activation='tanh', 
                                    kernel_initializer=xav_init,
                                    kernel_regularizer=l2_regularizer)(decoded)

    autoencoder = tf.keras.models.Model(input_data, decoded)
    autoencoder.compile(optimizer=optimize_adam, loss='mse')

    early = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, verbose=1, mode='min'
    )
    autoencoder.fit(X_train,X_train, epochs=nb_epoch, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test), callbacks=[early])
    return autoencoder

