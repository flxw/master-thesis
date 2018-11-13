import keras
import pickle
import random
import numpy as np
import pandas as pd
import re
import os

import tqdm
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Input, Reshape, concatenate, Flatten, Activation, LSTM
from keras.utils import np_utils

import multi_gpu_utils2 as multi_gpu_utils

##############################
##### CONFIGURATION SETUP ####
data_path = "../logs/bpic2011.xes"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
target_variable = "concept:name"
### CONFIGURATION SETUP END ###
###############################

def load_trace_dataset(purpose='categorical', ttype='test'):
    suffix = "_{0}_{1}.pickled".format(purpose, ttype)
    p = data_path.replace(".xes", suffix)
    return pickle.load(open(p, "rb"))

if __name__ == '__main__':    
    ### BE NICE SAY HELLO
    print("Welcome to Felix' master thesis: Deep Learning Next-Activity Prediction Using Subsequence-Enriched Input Data")
    print("Will now train a mimicked implementation of Schoenig's network as he has described it in his 2018 paper")
    print("\n")
    
    ### BEGIN DATA LOADING
    train_traces_categorical = load_trace_dataset('categorical', 'train')
    train_traces_ordinal = load_trace_dataset('ordinal', 'train')
    train_targets = load_trace_dataset('target', 'train')
    test_traces_categorical = load_trace_dataset('categorical', 'test')
    test_traces_ordinal = load_trace_dataset('ordinal', 'test')
    test_targets = load_trace_dataset('target', 'test')
    feature_dict = load_trace_dataset('mapping', 'dict')
    
    ### DO FINAL DATA PREPARATION
    # Use one-hot encoding for categorical values
    for col in train_traces_categorical[0].columns:
        nc = len(feature_dict[col]['to_int'].values())
        for i in range(0, len(train_traces_categorical)):
            tmp = train_traces_categorical[i][col].map(feature_dict[col]['to_int'])
            tmp = np_utils.to_categorical(tmp, num_classes=nc)
            tmp = pd.DataFrame(tmp).add_prefix(col)

            train_traces_categorical[i].drop(columns=[col], inplace=True)
            train_traces_categorical[i] = pd.concat([train_traces_categorical[i], tmp], axis=1)
    
    # tie everything together since we only have a single input layer
    train_traces = [ pd.concat([a,b], axis=1) for a,b in zip(train_traces_ordinal, train_traces_categorical)]
    n_train_cols  = len(train_traces[0].columns)
    n_target_cols = len(train_targets[0].columns)
    
    train_input_batches  = np.array([ t.values.reshape((-1,n_train_cols))  for t in train_traces ])
    train_target_batches = np.array([ t.values.reshape((-1,n_target_cols)) for t in train_targets])
    
    ### BEGIN MODEL CONSTRUCTION
    batch_size = None # None translates to unknown batch size
    output_count = len(feature_dict[target_variable]["to_int"])
    unit_count = n_train_cols + output_count
    # [samples, time steps, features]
    il = Input(batch_shape=(batch_size,None,n_train_cols))

    main_output = LSTM(unit_count,
                       batch_input_shape=(batch_size,None,n_train_cols),
                       stateful=False,
                       return_sequences=True,
                       dropout=0.3)(il)
    main_output = LSTM(unit_count,
                       stateful=False,
                       return_sequences=True,
                       dropout=0.3)(main_output)

    main_output = Dense(output_count, activation='softmax', name='dense_final')(main_output)
    full_model = Model(inputs=[il], outputs=[main_output])
    optimizerator = keras.optimizers.RMSprop()
    
#    full_model = multi_gpu_utils.multi_gpu_model(full_model)
    full_model.compile(loss='categorical_crossentropy', optimizer=optimizerator, metrics=['categorical_accuracy', 'mae'])
    
    ### BEGIN MODEL TRAINING
    n_epochs = 100
    best_acc = 0
    best_epoch = 0
    tr_acc_s = 0.0
    tr_loss_s = 0.0
    
    for epoch in range(1,n_epochs+1):
        mean_tr_acc  = []
        mean_tr_loss = []
        mean_tr_mae  = []
        
        for t_idx in tqdm.tqdm(range(0, len(train_input_batches)),
                               desc="Epoch {0}/{1} | {2:.2f}% | {3:.2f}".format(epoch,n_epochs, tr_acc_s, tr_loss_s)):
            
            # Each batch consists of a single sample, i.e. one whole trace (1)
            # A trace is represented by a variable number of timesteps (-1)
            # And finally, each timestep contains n_train_cols variables
            batch_x = train_input_batches[t_idx].reshape((1,-1,n_train_cols))
            batch_y = train_target_batches[t_idx].reshape((1,-1,n_target_cols))
            
            tr_loss, tr_acc, tr_mae = full_model.train_on_batch(batch_x, batch_y)
            mean_tr_acc.append(tr_acc)
            mean_tr_loss.append(tr_loss)
            mean_tr_mae.append(tr_mae)

        tr_acc_s = 100*round(np.mean(mean_tr_acc),3)
        tr_loss_s = np.mean(mean_tr_loss)

        if best_acc < tr_acc_s :
            best_acc = round(tr_acc_s, 3)
            best_epoch = epoch
            full_model.save('/home/felix.wolff2/docker_share/schoenig_baseline_e{0}_acc{1:.2f}.h5'.format(epoch,best_acc))
