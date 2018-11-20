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
if args.model == 'evermann':
    pass
elif args.model == 'schoenig':
    import schoenig_builder as model_builder
    n_epochs = 100
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
elif args.model == 'sp2':
    import sp2_builder as model_builder
    n_epochs = 150
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
elif args.model == 'pfs':
    pass
else:
    os.exit(0)

train_X, train_Y, test_X, test_Y, model = model_builder.prepare(args.datapath, target_variable)
statistics = pd.DataFrame(columns=['loss', 'acc', 'val_loss', 'val_acc', 'training_time', 'validation_time'], index=range(0,n_epochs), dtype=np.float32)

if args.mode == 'padded':
    # make cutoff step a function of the trace length in each percentile
    mlen = math.ceil(np.percentile([len(t) for t in train_Y], 80))
    
    # remove traces from training which are longer than mlen since they'll be removed anyway
    train_Y = list(filter(lambda t: len(t) <= mlen, train_Y))
    test_Y  = list(filter(lambda t: len(t) <= mlen, test_Y))
    
    for layer_name in test_X.keys():
        test_X[layer_name]  = list(filter(lambda t: len(t) <= mlen, test_X[layer_name]))
        train_X[layer_name] = list(filter(lambda t: len(t) <= mlen, train_X[layer_name]))
    
    # now pad all sequences to same length
    train_targets = pad_sequences(train_Y, padding='post')
    test_targets  = pad_sequences(test_Y,  padding='post')
    
    test_inputs = {}
    train_inputs = {}
    for layer_name in test_X.keys():
        test_inputs[layer_name]  = pad_sequences(test_X[layer_name],  padding='post')
        train_inputs[layer_name] = pad_sequences(train_X[layer_name], padding='post')
    
    ### BEGIN MODEL TRAINING
    batch_size = math.ceil(0.01*len(train_Y))
    early_stopper = EarlyStopping(monitor='val_loss',
                                                  min_delta=0,
                                                  patience=20,
                                                  verbose=1,
                                                  mode='auto',
                                                  baseline=None,
                                                  restore_best_weights=False)
    checkpointer = ModelCheckpoint("{0}/{1}/{2}/".format(remote_path, args.model, args.mode) + "model.{epoch:02d}-{val_loss:.2f}.hdf5",
                                                   monitor='val_loss',
                                                   verbose=1,
                                                   save_best_only=True,
                                                   save_weights_only=False,
                                                   mode='auto',
                                                   period=1)
    scb = StatisticsCallback(training_batchcount=math.ceil(len(train_inputs)/batch_size),
                             statistics_df=statistics,
                             accuracy_metric=model.metrics[0])

    history = model.fit(x=train_inputs,
                        y=train_targets,
                        validation_data=(test_inputs,test_targets),
                        batch_size=batch_size,
                        callbacks=[early_stopper,checkpointer,scb],
                        epochs=n_epochs,
                        verbose=1)


if args.mode == 'grouped':
    n_X_cols = train_X['seq_input'][0].shape[1]
    n_Y_cols = train_Y[0].shape[1]

    # loop through every dictionary key and group (since the elements had the same order before, they should have after)
    for input_name in train_X.keys():
        layer_train_X = train_X[input_name]
        grouped_train_X = {}
        grouped_train_Y = {}

        # create a dictionary entry for every timeseries length and put the traces in the appropriate bin
        for i in range(0,len(layer_train_X)):
            tl = len(layer_train_X[i])
            elX = np.array(layer_train_X[i])

            if tl in grouped_train_X:
                grouped_train_X[tl].append(elX)

                if (len(layer_train_X) == len(train_Y)):
                    grouped_train_Y[tl].append(train_Y[i])
            else:
                grouped_train_X[tl] = [elX]

                if (len(layer_train_X) == len(train_Y)):
                    grouped_train_Y[tl] = [train_Y[i]]

        train_X[input_name] = np.array([np.array(l) for l in grouped_train_X.values()])

        if (len(layer_train_X) == len(train_Y)):
            train_Y = np.array([np.array(l) for l in grouped_train_Y.values()])
    
    ### BEGIN TRAINING
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
        for batch_id in tqdm.trange(len(train_Y),
                                          desc="acc: {0:.2f} | loss: {1:.2f} | val_acc {2:.2f} | val_loss: {3:.2f}".format(np.mean(tr_accs),
                                                                                                                              np.mean(tr_losses),
                                                                                                                              np.mean(val_accs),
                                                                                                                              np.mean(val_losses))):
            # Each batch consists of a single sample, i.e. one whole trace (1)
            # A trace is represented by a variable number of timesteps (-1)
            # And finally, each timestep contains n_train_cols variables
            samples = len(train_X['seq_input'][batch_id])
            batch_y = train_Y[batch_id].reshape((samples,-1,n_Y_cols))
            batch_x = { layer_name:train_X[layer_name][batch_id] for layer_name in train_X.keys() }
            
            l,a = model.train_on_batch(batch_x, batch_y)
            tr_losses.append(l)
            tr_accs.append(a)
        training_time = time.time() - t_start
        
        # validating the epoch result
        t_start = time.time()
        for batch_id in range(len(test_Y)):
            batch_y = test_Y[batch_id].reshape((1,-1,n_Y_cols))
            batch_x = { layer_name:test_X[layer_name][batch_id].reshape((1,-1,n_X_cols)) for layer_name in test_X.keys() }
            
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
            print("Increased accuracy from {0:.2f}% to {1:.2f}% - saving model!".format(best_val_acc*100, last_val_acc*100))
            best_val_acc = last_val_acc
            model.save("{0}/{1}/{2}/best_val_acc_e{3}.hdf5".format(remote_path, args.model, args.mode, epoch))
    
statistics.to_pickle("{0}/{1}/{2}/train_statistics.pickled".format(remote_path, args.model, args.mode))
