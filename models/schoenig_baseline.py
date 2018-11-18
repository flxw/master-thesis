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
remote_path = "/home/felix.wolff2/docker_share"
data_path = "../logs/normalized/bpic2011.xes"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
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
    train_traces_targets = load_trace_dataset('target', 'train')
    test_traces_categorical = load_trace_dataset('categorical', 'test')
    test_traces_ordinal = load_trace_dataset('ordinal', 'test')
    test_traces_targets = load_trace_dataset('target', 'test')
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
        for i in range(0, len(test_traces_categorical)):
            tmp = test_traces_categorical[i][col].map(feature_dict[col]['to_int'])
            tmp = np_utils.to_categorical(tmp, num_classes=nc)
            tmp = pd.DataFrame(tmp).add_prefix(col)

            test_traces_categorical[i].drop(columns=[col], inplace=True)
            test_traces_categorical[i] = pd.concat([test_traces_categorical[i], tmp], axis=1)
    
    # tie everything together since we only have a single input layer
    train_traces = [ pd.concat([a,b], axis=1) for a,b in zip(train_traces_ordinal, train_traces_categorical)]
    test_traces  = [ pd.concat([a,b], axis=1) for a,b in zip(test_traces_ordinal, test_traces_categorical)]
    n_train_cols  = len(train_traces[0].columns)
    n_target_cols = len(train_traces_targets[0].columns)
    mlen = int(np.mean([len(t) for t in train_traces]) * 1.25)

    train_inputs  = keras.preprocessing.sequence.pad_sequences(train_traces, maxlen=mlen, padding='pre')
    train_targets = keras.preprocessing.sequence.pad_sequences(train_traces_targets, maxlen=mlen, padding='pre')
    assert(len(train_inputs) == len(train_targets))
    assert(sum([len(t) for t in train_inputs]) == sum([len(t) for t in train_targets]))
    assert(sum([len(t) for t in train_inputs]) == sum([len(t) for t in train_targets]))
    
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
    
    full_model = multi_gpu_utils.multi_gpu_model(full_model)
    full_model.compile(loss='categorical_crossentropy', optimizer=optimizerator, metrics=['accuracy'])
    
    ### BEGIN MODEL TRAINING
    n_epochs = 100
    early_stopper = keras.callbacks.EarlyStopping(monitor='loss',
                                                  min_delta=0,
                                                  patience=20,
                                                  verbose=1,
                                                  mode='auto',
                                                  baseline=None,
                                                  restore_best_weights=False)
    checkpointer = keras.callbacks.ModelCheckpoint(remote_path + "/schoenig_weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                                   monitor='loss',
                                                   verbose=1,
                                                   save_best_only=True,
                                                   save_weights_only=False,
                                                   mode='auto',
                                                   period=1)

    history = full_model.fit(x=train_inputs,
                             y=train_targets,
                             validation_data=(test_traces,test_traces_targest),
                             batch_size=32,
                             epochs=100,
                             verbose=2)
