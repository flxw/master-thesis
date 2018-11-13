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

import multi_gpu_utils2 as multi_gpu_utils

##############################
##### CONFIGURATION SETUP ####
data_path = "../logs/bpic2011.xes"
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    print("Will now train a mimicked implementation of Evermann's network as he has described it in his 2016 paper")
    print("\n")
    
    ### BEGIN DATA LOADING
    train_traces = load_trace_dataset('categorical', 'train')
    train_targets = load_trace_dataset('target', 'train')
    test_traces = load_trace_dataset('categorical', 'test')
    test_targets = load_trace_dataset('target', 'test')
    feature_dict = load_trace_dataset('mapping', 'dict')
    
    ### BEGIN MODEL CONSTRUCTION
    batch_size = None # None translates to unknown batch size
    # [samples, time steps, features]
    il = Input(batch_shape=(batch_size,None,1))
    main_output = Embedding(624, 500)(il)
    main_output = Reshape(target_shape=(-1,500))(main_output) # reshape layer does not need to know BATCH SIZE!!!

    # sizes should be multiple of 32 since it trains faster due to np.float32
    main_output = LSTM(500,
                       batch_input_shape=(batch_size,None,1),
                       stateful=False,
                       return_sequences=True,
                       unroll=False,
                       dropout=0.2,
                       kernel_initializer=keras.initializers.Zeros())(main_output)
    main_output = LSTM(500,
                       stateful=False,
                       return_sequences=True,
                       unroll=False,
                       dropout=0.2,
                       kernel_initializer=keras.initializers.Zeros())(main_output)

    main_output = Dense(len(feature_dict[target_variable]["to_int"]), activation='softmax', name='dense_final')(main_output)
    full_model = Model(inputs=[il], outputs=[main_output])
    optimizerator = keras.optimizers.SGD(lr=1)
    
    full_model = multi_gpu_utils.multi_gpu_model(full_model)
    full_model.compile(loss='categorical_crossentropy', optimizer=optimizerator, metrics=['categorical_accuracy', 'mae'])
    
    ### DO FINAL DATA PREPARATION
    train_traces = [ t[target_variable].map(feature_dict[target_variable]['to_int']).values for t in train_traces ]
    train_input_batches  = np.array([ t.reshape((-1,1)) for t in train_traces ])
    train_target_batches = np.array([ t.values.reshape((-1,625)) for t in train_targets])
    
    ### BEGIN MODEL TRAINING
    n_epochs = 50
    best_acc = 0
    tr_acc = 0.0
    
    for epoch in range(1,n_epochs+1):
        mean_tr_acc  = []
        mean_tr_loss = []
        mean_tr_mae  = []
        
        for t_idx in tqdm.tqdm(range(0, len(train_input_batches)),
                               desc="Epoch {0}/{1} | Last accuracy {2:.2f}%".format(epoch,n_epochs, tr_acc)):
            batch_x = train_input_batches[t_idx].reshape((1,-1,1))
            batch_y = train_target_batches[t_idx].reshape((1,-1,625))
            
            tr_loss, tr_acc, tr_mae = full_model.train_on_batch(batch_x, batch_y)
            mean_tr_acc.append(tr_acc)
            mean_tr_loss.append(tr_loss)
            mean_tr_mae.append(tr_mae)

        if epoch == 25: # why? See Implementation in evermann2016
            keras.backend.set_value(full_model.optimizer.decay, .75)

        tr_acc = 100*round(np.mean(mean_tr_acc),3)
        if best_acc < tr_acc:
            best_acc = tr_acc
            
            if epoch > 20:
                full_model.save('/home/felix.wolff2/docker_share/evermann_baseline_e{0}_acc{1:.3f}.h5'.format(epoch,best_acc))
