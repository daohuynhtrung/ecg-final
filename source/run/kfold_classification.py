import argparse
import json
import shutil
from os import path
import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from source import utils
from source.data.DataHelper import k_fold_single_rr_data_pipeline
from source.models.models import lstm_classifier

def train():
    parser = argparse.ArgumentParser(description='NA')
    parser.add_argument('-c', '--configure', default='configure/classifier.json', help='JSON file')
    args = parser.parse_args()

    with open(args.configure) as f:
        config = json.load(f)

    acc = []
    loss = []
    val_acc = []
    val_loss = []
    score = []
    fold = 0

    #Load function from json
    load_data_str = config.get('load_data_fn')
    load_data_fn, mod_name = utils.load_func_by_name(load_data_str)
    load_model_str = config.get('model')
    load_model, mod_name_model = utils.load_func_by_name(load_model_str)

    #create checkpoint path
    checkpoint_path = utils.make_dir_epoch_time(config['checkpoint'])

    # copy configure file to reference later
    shutil.copy(args.configure, checkpoint_path)

    # load data
    for X_train, Y_train, X_test, Y_test in k_fold_single_rr_data_pipeline(config['data_path'], config['k'],
                                                                           load_data_fn=load_data_fn):
        # expand dim to [?, 1, vec_size] for LSTM
        X_train = np.expand_dims(X_train, axis=1)
        X_test = np.expand_dims(X_test, axis=1)
        
        # create model
        model = load_model(**config)

        # checkpoint
        filepath = checkpoint_path + "/" + "cls-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        hist = model.fit(x=X_train, y=Y_train,
                         batch_size=config['batch_size'],
                         epochs=config['epochs'],
                         validation_data=(X_test, Y_test),
                         shuffle=True,
                         callbacks=callbacks_list
                        )
        
        Y_pred = model.predict(X_test)
        Y_pred = (Y_pred > 0.5)
        Y_pred = np.argmax(Y_pred, axis=1)
        Y_test = np.argmax(Y_test, axis=1)
        score.append(accuracy_score(Y_test, Y_pred))

        acc.append(hist.history['acc'])
        loss.append(hist.history['loss'])
        val_acc.append(hist.history['val_acc'])
        val_loss.append(hist.history['val_loss'])
        
        #Plot result and save
        # utils.plot_result(hist, config, fold)
        fold += 1
    
    #result
    print('Saving result...')
    utils.mean_result(acc, loss, val_acc, val_loss, score, config, checkpoint_path)
    print('Result saved !')

######################
if __name__ == "__main__":
    train()
else:
    print("classification is being imported into another module")
