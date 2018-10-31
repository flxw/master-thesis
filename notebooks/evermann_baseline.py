import argparse
parser = argparse.ArgumentParser(description='Train a mimicked implementation of Jorg Evermann\'s neural network')
parser.add_argument('--continue', dest='network_to_continue', action='store', default=False,
                     help='Which network file to continue training')
args = parser.parse_args()

import keras
import pickle
import random
import numpy as np
import pandas as pd
import re
import os

from tqdm import *
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Input, Reshape, concatenate, Flatten, Activation, LSTM
from keras.utils import np_utils

##############################
##### CONFIGURATION SETUP ####
data_path = "../logs/bpic2011.xes"
traces_finalpath = data_path.replace(".xes", "_traces_encoded.pickled")
traces_dictionarypath = data_path.replace(".xes", "_dictionaries.pickled")
target_variable = "concept:name"

traces = pickle.load(open(traces_finalpath, "rb"))
feature_dict = pickle.load(open(traces_dictionarypath, "rb"))
windowsize = 20
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
### CONFIGURATION SETUP END ###
###############################

if __name__ == '__main__':    
    # be nice, say hello
    print("Welcome to Felix' master thesis: Deep Learning Next-Activity Prediction Using Subsequence-Enriched Input Data")
    print("Will now train a mimicked implementation of Evermann's network as he has described it in his 2016 paper")
#     print("Total number of GPUs to be used: {0}".format(ngpus))
    print("\n")
    # shuffle complete traces and create test and training set
    random.shuffle(traces)
    sep_idx = int(0.8*len(traces))
    
    batch_size = 1
    # [samples, time steps, features]
    il = Input(batch_shape=(batch_size,None,1))
    main_output = il
    # main_output = Embedding(624, 500)(il)

    # sizes should be multiple of 32 since it trains faster due to np.float32
    main_output = LSTM(500,
                       batch_input_shape=(batch_size,None,500),
                       stateful=True,
                       return_sequences=True,
                       unroll=False,
                       kernel_initializer=keras.initializers.glorot_uniform(seed=123))(main_output)
    # main_output = LSTM(500,
    #                    stateful=False,
    #                    return_sequences=False,
    #                    kernel_initializer=keras.initializers.glorot_uniform(seed=123),
    #                    activation='sigmoid')(main_output)

    main_output = Dense(len(feature_dict["concept:name"]["to_int"]), activation='softmax', name='dense_final')(main_output)

    full_model = Model(inputs=[il], outputs=[main_output])
    optimizerator = keras.optimizers.adam()
    full_model.compile(loss='categorical_crossentropy', optimizer=optimizerator, metrics=['accuracy'])
    
    if args.network_to_continue:
        full_model = keras.models.load_model(args.network_to_continue)
        
    n_epochs = 100
    for epoch in range(1,n_epochs+1):
        mean_tr_acc  = []
        mean_tr_loss = []
        
        for t_idx in tqdm(range(sep_idx, len(traces), batch_size), desc="Epoch {0}/{1}".format(epoch,n_epochs)):
            traces_batch = traces[t_idx:t_idx+batch_size]
            batch_x = np.array([ t["concept:name"].values.reshape((-1,1)) for t in traces_batch ])
            batch_y = np.array([ np_utils.to_categorical(t["TARGET"].values, num_classes=625) for t in traces_batch ])

            tr_loss, tr_acc = full_model.train_on_batch(batch_x, batch_y)
            mean_tr_acc.append(tr_acc)
            mean_tr_loss.append(tr_loss)

        print('Epoch {0} -- loss = {1:.5f} -- acc = {2:.5f}'.format(epoch,np.mean(mean_tr_loss), np.mean(mean_tr_acc)))
        full_model.save('evermann_baseline_e{0}.h5'.format(epoch))  # creates a HDF5 file 'my_model.h5'
