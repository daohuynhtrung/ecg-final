import argparse
import json
import shutil
from os import path
import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error 
import utils
from data.data_stuff import data_pipeline
from contextlib import redirect_stdout

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
    load_model_str = config.get('model')
    load_model, mod_name_model = utils.load_func_by_name(load_model_str)
    load_data_str = config.get('data_processing')
    load_data, mod_name_data = utils.load_func_by_name(load_data_str)


    #create checkpoint path
    checkpoint_path = utils.make_dir_epoch_time(config['checkpoint'])

    # copy configure file to reference later
    shutil.copy(args.configure, checkpoint_path)

    # load data
    X_train, Y_train, X_test, Y_test = load_data(config['data_path'])
    # expand dim to [?, 1, vec_size] for LSTM
    # X_train = np.expand_dims(X_train, axis=1)
    # X_test = np.expand_dims(X_test, axis=1)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # create model
    model = load_model(**config)
    
    # checkpoint
    filepath = checkpoint_path + "/" + "cls-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    callbacks_list = [es, checkpoint]


    hist = model.fit(x=X_train, y=Y_train,
                        batch_size=config['batch_size'],
                        epochs=config['epochs'],
                        validation_data=(X_test, Y_test),
                        shuffle='True',
                        callbacks=callbacks_list
                    )
    # model summary
    with open(checkpoint_path+'/summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary() 
    # accuracy = model.evaluate(X_test, Y_test, verbose=0)
    
    # Y_pred = (Y_pred > 0.5)
    # Y_pred = np.argmax(Y_pred, axis=1)
    # Y_test = np.argmax(Y_test, axis=1)
    # score.append(accuracy_score(Y_test, Y_pred))
    

    y_pred = model.predict(X_test, batch_size=1024)

    y_pred = np.argmax(y_pred, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    f1 = classification_report(Y_test, y_pred)
    print('Accuracy: ',accuracy_score(Y_test, y_pred))
    print(f1)
    
    # Save f1 score
    json.dump(f1,open(checkpoint_path+'/f1.json','w')) 
    
    history_dict = hist.history
    # Save result
    json.dump(history_dict, open(checkpoint_path+'/result.json','w'))

    # Save fig
    utils.plot_result_by_history(history_dict, checkpoint_path)

    #Save difference result
    utils.save_dif_result(y_pred, Y_test, X_test, checkpoint_path)
######################
if __name__ == "__main__":
    train()
else:
    print("classification is being imported into another module")
