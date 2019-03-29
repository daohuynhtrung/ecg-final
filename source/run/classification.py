import argparse
import json
import shutil
from os import path
import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error 
from source import utils
from source.data.data_stuff import data_pipeline

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
    X_train, Y_train, X_test, Y_test = data_pipeline(config['data_path'], config['k'],load_data_fn=load_data_fn)
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
    # model summary
    # print(model.summary())

    # accuracy = model.evaluate(X_test, Y_test, verbose=0)
    
    # Y_pred = (Y_pred > 0.5)
    # Y_pred = np.argmax(Y_pred, axis=1)
    # Y_test = np.argmax(Y_test, axis=1)
    # score.append(accuracy_score(Y_test, Y_pred))
    

    y_pred = model.predict(X_test, batch_size=1024)
    print(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    print(y_pred)
    print(Y_test)

    print('Accuracy: ',accuracy_score(Y_test, y_pred))
    # print('Orther estimate: ',precision_recall_fscore_support(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))
    
    #result
    # print('Saving result...')
    # utils.mean_result(acc, loss, val_acc, val_loss, score, config, checkpoint_path)
    # print('Result saved !')

######################
if __name__ == "__main__":
    train()
else:
    print("classification is being imported into another module")
