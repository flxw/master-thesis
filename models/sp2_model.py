import tensorflow as tf
import keras
import pickle
import random
import numpy as np
import pandas as pd
import re
import os
import itertools
from tqdm import tqdm
import json

from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Input, Reshape, concatenate, ReLU, Activation, LSTM, Dropout
from keras.utils import np_utils

def write_log(callback, names, logs, batch_no, tuning_parameters):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

##############################
##### CONFIGURATION SETUP ####
log_path  = "/home/felix.wolff2/docker_share/sp2_logs"
data_path = "../logs/normalized/bpic2011.xes"
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
target_variable = "concept:name"
stop_patience=20
stop_delta=0.01
### CONFIGURATION SETUP END ###
###############################

def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

def load_trace_dataset(purpose='categorical', ttype='test'):
    suffix = "_{0}_{1}.pickled".format(purpose, ttype)
    p = data_path.replace(".xes", suffix)
    return pickle.load(open(p, "rb"))
    
def sp2_model(train_input_batches_seq, train_input_batches_sp2, train_target_batches, 
              test_input_batches_seq, test_input_batches_sp2, test_target_batches, params):
    ### BEGIN MODEL CONSTRUCTION
    batch_size = None # None translates to unknown batch size
    seq_unit_count = n_seq_cols + n_target_cols
    sp2_unit_count = n_sp2_cols + n_target_cols

    # array format: [samples, time steps, features]
    il = Input(batch_shape=(batch_size,None,n_seq_cols), name="seq_input")

    # sizes should be multiple of 32 since it trains faster due to np.float32
    main_output = LSTM(seq_unit_count,
                       batch_input_shape=(batch_size,None,n_seq_cols),
                       stateful=False,
                       return_sequences=True,
                       unroll=False,
                       kernel_initializer=keras.initializers.glorot_normal(),
                       dropout=params['dropout'])(il)
    main_output = LSTM(seq_unit_count,
                       stateful=False,
                       return_sequences=True,
                       unroll=False,
                       kernel_initializer=keras.initializers.glorot_normal(),
                       dropout=params['dropout'])(main_output)

    # SP2 input here
    il2 = Input(batch_shape=(batch_size,None,n_sp2_cols), name="sec_input")
    sp2 = Dense(sp2_unit_count, activation='relu')(il2)
    
    main_output = concatenate([main_output, sp2], axis=-1)
    main_output = Dropout(params['dropout'])(main_output)
    main_output = Dense(n_target_cols, activation='relu')(main_output)
    main_output = Dropout(params['dropout'])(main_output)
    main_output = Dense(n_target_cols, activation='softmax')(main_output)
    # add softmax activation for classification and play with relu for hidden layers

    full_model = Model(inputs=[il, il2], outputs=[main_output])

    full_model.compile(loss='categorical_crossentropy',
                       optimizer=params['optimizer'],
                       metrics=['categorical_accuracy'])
    
    ### SET UP MODEL LOGGING
    callback = keras.callbacks.TensorBoard(log_path)
    callback.set_model(full_model)
    
    ### BEGIN MODEL TRAINING
    n_epochs = 150
    best_acc = 0
    tr_acc_s = 0.0
    tr_loss_s = 0
    last_loss = 0.0
    patience_counter = 0
    
    for epoch in range(1,n_epochs+1):
        mean_tr_acc  = []
        mean_tr_loss = []
        
        for t_idx in tqdm(range(0, len(train_input_batches_seq)),
                               desc="Epoch {0}/{1} | Last accuracy {2:.2f}% | Last loss: {3:.2f}".format(epoch,n_epochs, tr_acc_s, tr_loss_s)):
            
            # Each batch consists of a single sample, i.e. one whole trace (1)
            # A trace is represented by a variable number of timesteps (-1)
            # And finally, each timestep contains n_train_cols variables
            batch_x_seq = train_input_batches_seq[t_idx].reshape((1,-1,n_seq_cols))
            batch_x_sp2 = train_input_batches_sp2[t_idx].reshape((1,-1,n_sp2_cols))
            batch_y = train_target_batches[t_idx].reshape((1,-1,n_target_cols))
            
            tr_loss, tr_acc = full_model.train_on_batch({'seq_input': batch_x_seq, 'sec_input': batch_x_sp2}, batch_y)
            mean_tr_acc.append(tr_acc)
            mean_tr_loss.append(tr_loss)
            
            # Log results from batch
            write_log(callback, ['train_loss','train_acc'], (tr_loss,tr_acc), t_idx, params)
            
        tr_acc_s = 100*round(np.mean(mean_tr_acc),3)
        tr_loss_s = np.mean(mean_tr_loss)

        if best_acc < tr_acc_s:
            best_acc = tr_acc_s
            
        if stop_delta > (last_loss-tr_loss_s):
            patience_counter+=1
        else:
            patience_counter = 0
            
        if patience_counter == stop_patience:
            tqdm.write("Reached early-stopping threshold!")
            break;
            
        last_loss = tr_loss_s

    return best_acc, full_model

if __name__ == '__main__':    
    ### BE NICE SAY HELLO
    print("Welcome to Felix' master thesis: Deep Learning Next-Activity Prediction Using Subsequence-Enriched Input Data")
    print("Will now tune hyper-parameters for the SP2 model on the BPIC2011 dataset")
    print("\n")
    
    ### BEGIN DATA LOADING
    train_traces_categorical = load_trace_dataset('categorical', 'train')
    train_traces_ordinal = load_trace_dataset('ordinal', 'train')
    train_targets = load_trace_dataset('target', 'train')
    train_traces_sp2 = load_trace_dataset('sp2', 'train')

    test_traces_categorical = load_trace_dataset('categorical', 'test')
    test_traces_ordinal = load_trace_dataset('ordinal', 'test')
    test_targets = load_trace_dataset('target', 'test')
    test_traces_sp2 = load_trace_dataset('sp2', 'test')

    feature_dict = load_trace_dataset('mapping', 'dict')
    
    ### DO FINAL DATA PREPARATION
    # Use one-hot encoding for categorical values in training and test set
    for col in train_traces_categorical[0].columns:
        nc = len(feature_dict[col]['to_int'].values())
        for i in range(0, len(train_traces_categorical)):
            tmp = train_traces_categorical[i][col].map(feature_dict[col]['to_int'])
            tmp = np_utils.to_categorical(tmp, num_classes=nc)
            tmp = pd.DataFrame(tmp).add_prefix(col)

            train_traces_categorical[i].drop(columns=[col], inplace=True)
            train_traces_categorical[i] = pd.concat([train_traces_categorical[i], tmp], axis=1)
            
        for i in range(0, len(test_traces_categorical)):
            tmp = test_traces_categorical[i][col].map(feature_dict[col]['to_int'])
            tmp = np_utils.to_categorical(tmp, num_classes=nc)
            tmp = pd.DataFrame(tmp).add_prefix(col)

            test_traces_categorical[i].drop(columns=[col], inplace=True)
            test_traces_categorical[i] = pd.concat([test_traces_categorical[i], tmp], axis=1)
            
    # categorical and ordinal inputs are fed in on one single layer
    train_traces_seq = [ pd.concat([a,b], axis=1) for a,b in zip(train_traces_ordinal, train_traces_categorical) ]
    test_traces_seq  = [ pd.concat([a,b], axis=1) for a,b in zip(test_traces_ordinal,  test_traces_categorical)  ]
    
    # tie everything together since we only have a single input layer
    n_sp2_cols = len(train_traces_sp2[0].columns)
    n_seq_cols = len(train_traces_seq[0].columns)
    n_target_cols = len(train_targets[0].columns)
    
    train_input_batches_seq  = np.array([ t.values  for t in train_traces_seq ])
    train_input_batches_sp2  = np.array([ t.values  for t in train_traces_sp2 ])
    train_target_batches     = np.array([ t.values for t in train_targets])
    
    test_input_batches_seq  = np.array([ t.values for t in test_traces_seq ])
    test_input_batches_sp2  = np.array([ t.values for t in test_traces_sp2 ])
    test_target_batches     = np.array([ t.values for t in test_targets ])
    
    
    ### DEFINE HYPER-PARAMETER TUNING RANGE
    params = {
        'optimizer': ['adagrad', 'rmsprop', 'adam' ],
        'dropout': [0.1, 0.3, 0.5]
    }
    
    # all the hyperoptimization libraries suck if you do not use fit
    # just do a simple self-built version here
    winner_acc = 0
    winner_model = 0
    winner_params = None
    for param_combo in dict_product(params):
        print("Testing hyperparameter combination:", param_combo)
        acc,model = sp2_model(train_input_batches_seq,
                  train_input_batches_sp2,
                  train_target_batches, 
                  test_input_batches_seq,
                  test_input_batches_sp2,
                  test_target_batches,
                  param_combo)
        
        if acc > winner_acc:
            winner_acc = acc
            winner_model = model
            winner_params = param_combo
            
    winner_model.save('/home/felix.wolff2/docker_share/sp2_hypertuning_winner_acc{0:.2f}.h5'.format(winner_acc))
    pickle.dump(winner_params, open("/home/felix.wolff2/docker_share/sp2_hypertuning_winner_params", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    
    #print(winner_model.evaluate({'seq_input': test_input_batches_seq, 'sp2_input': test_input_batches_sp2}, test_target_batches, batch_size=1))
    print("Best performing model chosen hyper-parameters:", winner_params)
