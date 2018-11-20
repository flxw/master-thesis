import argparse
import os
import tqdm
import time
import numpy as np
import pandas as pd
import pickle
import math

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from utils import StatisticsCallback

parser = argparse.ArgumentParser(description='The network training script for Felix Wolff\'s master\'s thesis!')
parser.add_argument('model', choices=('evermann', 'schoenig', 'sp2', 'pfs'),
                    help='Which type of model to train.')
parser.add_argument('mode', choices=('padded', 'grouped', 'individual'),
                    help='Which mode to use for feeding the data into the model.')
parser.add_argument('datapath', help='Path of dataset to use for training.')

args = parser.parse_args()
args.datapath = os.path.abspath(args.datapath)

# TODO make all model scripts into simple plugin helpers for this master script which commands the training logic
# TODO load model depending on choice, build inside model script
# TODO implement data loading function within individual model scripts
# TODO implement grouping and padding handling here, also for the bipartite models
#      --> make handling easier: (simply make every output of data function a dictionary)
# TODO do all of training in a notebook to make result graphing easier (:

### CONFIGURATION
n_epochs = None
remote_path = "/home/felix.wolff2/docker_share"
target_variable = "concept:name"
### END CONFIGURATION


### BEGIN ################################################################
if args.model == 'schoenig':
    import schoenig_builder as model_builder
    n_epochs = 100
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# TODO import evermann etc

train_X, train_Y, test_X, test_Y, model = model_builder.prepare(args.datapath, target_variable)
statistics = pd.DataFrame(columns=['loss', 'acc', 'val_loss', 'val_acc', 'training_time', 'validation_time'], index=range(0,n_epochs), dtype=np.float32)

if args.mode == 'padded':
    # make cutoff step a function of the trace length in each percentile
    mlen = math.ceil(np.percentile([len(t) for t in train_X], 80))
    
    # remove traces from training which are longer than mlen since they'll be removed anyway
    train_X = list(filter(lambda t: len(t) <= mlen, train_X))
    train_Y = list(filter(lambda t: len(t) <= mlen, train_Y))
    test_X  = list(filter(lambda t: len(t) <= mlen, test_X))
    test_Y  = list(filter(lambda t: len(t) <= mlen, test_Y))
    
    # now pad all sequences to same length
    train_inputs  = pad_sequences(train_X, padding='pre')
    train_targets = pad_sequences(train_Y, padding='pre')
    test_inputs   = pad_sequences(test_X,  padding='pre')
    test_targets  = pad_sequences(test_Y,   padding='pre')
    
    ### BEGIN MODEL TRAINING
    batch_size = 32 # TODO: TUNE!!
    early_stopper = EarlyStopping(monitor='val_loss',
                                                  min_delta=0,
                                                  patience=20,
                                                  verbose=1,
                                                  mode='auto',
                                                  baseline=None,
                                                  restore_best_weights=False)
    checkpointer = ModelCheckpoint("{0}/{1}/{2}/".format(remote_path, args.model, args.mode) + "weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                                   monitor='val_loss',
                                                   verbose=1,
                                                   save_best_only=True,
                                                   save_weights_only=False,
                                                   mode='auto',
                                                   period=1)
    cb = StatisticsCallback(training_batchcount=math.ceil(len(train_inputs)/batch_size), statistics_df=statistics)

    history = model.fit(x=train_inputs,
                        y=train_targets,
                        validation_data=(test_inputs,test_targets),
                        batch_size=batch_size,
                        callbacks=[early_stopper,checkpointer],
                        epochs=n_epochs,
                        verbose=1)


if args.mode == 'grouped':
    n_X_cols = train_X[0].shape[1]
    n_Y_cols = train_Y[0].shape[1]

    grouped_train_X = {}
    grouped_train_Y = {}

    # create a dictionary entry for every timeseries length and put the traces in the appropriate bins
    for i in range(0,len(train_X)):
        tl = len(train_X[i])
        
        if tl in grouped_train_X:
            np.append(grouped_train_X[tl], train_X[i])
            np.append(grouped_train_Y[tl], train_Y[i])
        else:
            grouped_train_X[tl] = np.array([train_X[i]])
            grouped_train_Y[tl] = np.array([train_Y[i]])
        
    grouped_train_X = list(grouped_train_X.values())
    grouped_train_Y = list(grouped_train_Y.values())
    
    ### BEGIN TRAINING
    #        batch_statistics = np.array([], dtype=[(colname, np.float32) for colname in (model.loss + model.metrics)])    
    last_acc = 0
    last_loss = 0
    last_val_acc = 0
    last_val_loss = 0
    best_val_acc = 0

    for epoch in tqdm.tqdm(range(1, n_epochs+1), desc="Top Accuracy: {0:.2f}%".format(best_val_acc*100), leave=False):
        tr_accs = [0]
        tr_losses = [0]
        val_accs = [0]
        val_losses = [0]
        
        # training an epoch
        t_start = time.time()
        for batch_x, batch_y in tqdm.tqdm(zip(grouped_train_X, grouped_train_Y),
                                          desc="acc: {0:.2f}% | loss: {1:.2f}% | val_acc {2:.2f}% | val_loss: {3:.2f}".format(np.mean(tr_accs)*100,
                                                                                                                              np.mean(tr_losses),
                                                                                                                              np.mean(val_accs)*100,
                                                                                                                              np.mean(val_losses))):
            # Each batch consists of a single sample, i.e. one whole trace (1)
            # A trace is represented by a variable number of timesteps (-1)
            # And finally, each timestep contains n_train_cols variables
            l,a = model.train_on_batch(batch_x, batch_y)
            tr_losses.append(l)
            tr_accs.append(a)
        training_time = time.time() - t_start
        
        # validating the epoch result
        t_start = time.time()
        for batch_x, batch_y in zip(test_X, test_Y):
            batch_x = batch_x.reshape((1,-1,n_X_cols))
            batch_y = batch_y.reshape((1,-1,n_Y_cols))
            
            l,a = model.evaluate(x=batch_x, y=batch_y, batch_size=1, verbose=0)
            val_losses.append(l)
            val_accs.append(a)
        validation_time = time.time() - t_start
        
        statistics.values[epoch-1] = [np.mean(tr_losses),
                                      np.mean(tr_accs),
                                      np.mean(val_losses),
                                      np.mean(val_accs),
                                      training_time,
                                      validation_time]
        
        if best_val_acc < last_val_acc:
            best_val_acc = last_val_acc
            model.save("{0}/{1}/{2}/best_val_acc_e{3}.hdf5".format(remote_path, args.model, args.mode, epoch))
    
statistics.to_pickle("{0}/{1}/{2}/train_statistics.pickled".format(remote_path, args.model, args.mode))
