import keras
import pickle
import random
import numpy as np
import pandas as pd
import re
import multiprocessing
import os

from tqdm import *
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Input, Reshape, concatenate, Flatten, Activation, LSTM

def generate_input_name(var_name):
    return "input_{0}".format(''.join(c for c in var_name if c.isalnum()))
    
def wrapped__create_learning_dicts_from_trace(p):
    return create_learning_dicts_from_trace(*p)

def create_learning_dicts_from_trace(t, sp2_col_start_index, n_sp2_features, pfs_col_start_index, n_pfs_features, target_col_start_index, feature_names):
    t_dict = {'x':[], 'y':[]}
    # generate one input sequence for every type of variable
    # map every single-step batch in a dictionary that will correspond to input layer names!
    for i in range(0, len(t)):
        batch_dict = {}

        # automatically run through all ordinal and categorical features
        for col_idx, col in enumerate(feature_names[:sp2_col_start_index]):
            input_name = generate_input_name(col)
            batch_dict[input_name] = np.array(t.iloc[i, col_idx], dtype=np.float32).reshape([-1,1])

        # create batches for sp2 and pfs2 seperately because of their variable encodings
        batch_dict[generate_input_name("sp2")] = np.asarray(t.iloc[i, sp2_col_start_index:pfs_col_start_index], dtype=np.float32).reshape([-1,n_sp2_features])
        batch_dict[generate_input_name("pfs")] = np.asarray(t.iloc[i, pfs_col_start_index:target_col_start_index], dtype=np.float32).reshape([-1,n_pfs_features])

        t_dict['x'].append(batch_dict)
    t_dict['y'] = keras.utils.np_utils.to_categorical(t.iloc[:, target_col_start_index:].values.reshape([-1,1,1]))
    return t_dict

##############################
##### CONFIGURATION SETUP ####
data_path = "../logs/bpic2011.xes"
traces_finalpath = data_path.replace(".xes", "_traces_encoded.pickled")
traces_dictionarypath = data_path.replace(".xes", "_dictionaries.pickled")
n_sp2_features = 624
n_pfs_features = 25

traces = pickle.load(open(traces_finalpath, "rb"))
feature_dict = pickle.load(open(traces_dictionarypath, "rb"))
ncores = multiprocessing.cpu_count()
windowsize = 20
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
### CONFIGURATION SETUP END ###
###############################

if __name__ == '__main__':
    # be nice, say hello
    print("Welcome to Felix' master thesis: Deep Learning Next-Activity Prediction Using Subsequence-Enriched Input Data")
    print("Will now train a mimicked implementation of Schoenig's network as he has described it in his 2016 paper")
#     print("Total number of GPUs to be used: {0}".format(ngpus))
    print("\n")
    # shuffle complete traces and create test and training set
    random.shuffle(traces)
    sep_idx = int(0.8*len(traces))

    # extract the feature indices
    # data is organized like this: ordinal features | categorical features | SP2 features | PFS features | TARGET features
    # needed as every of these features will get its own layer
    feature_names  = traces[0].columns
    trace_columns = list(map(lambda e: bool(re.match('^TARGET$', e)), feature_names))
    target_col_start_index = trace_columns.index(True)

    categorical_feature_names = feature_dict.keys()
    pfs_col_start_index = target_col_start_index - n_pfs_features
    sp2_col_start_index = pfs_col_start_index - n_sp2_features
    cat_col_start_index = sp2_col_start_index - len(categorical_feature_names)

    ordinal_feature_names = feature_names[0:cat_col_start_index]

    models = []
    model_inputs = []

    ### BEGIN MODEL CONSTRUCTION
    il = Input(batch_shape=(1,sp2_col_start_index-1))
    main_output = lst(il)
    main_output = lstm(main_output)
    main_output = dense(main_output)
    # TODO: Schönig uses 2 LSTM layers with dropout=0.3, rmsprop optimization, stateful=False is what he says
    # TODO: separate output layer with softmax activation function
    

    # after LSTM has learned on the sequence, bring in the SP2/PFS features, like in Shibatas paper
    main_output = concatenate([main_output, sp2_embedding])
    main_output = Dense(20*32, activation='relu', name='dense_join')(main_output)
    main_output = Dense(len(feature_dict["concept:name"]["to_int"]), activation='sigmoid', name='dense_final')(main_output)

    full_model = Model(inputs=model_inputs, outputs=[main_output])
    full_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['categorical_accuracy', 'mae'])

    print("===> Beginning training....")
    
    n_epochs = 12
    for epoch in range(n_epochs):
        mean_tr_acc  = []
        mean_tr_loss = []
        mean_tr_mae  = []

        for t in tqdm(train_traces, desc="Epoch {0}/{1}".format(epoch,n_epochs)):
            for x,y in zip(t['x'],t['y']):
                tr_loss, tr_acc, tr_mae = full_model.train_on_batch(x, y)
                mean_tr_acc.append(tr_acc)
                mean_tr_loss.append(tr_loss)
                mean_tr_mae.append(tr_mae)
            full_model.reset_states()

        print('Epoch {0} -- loss = {1} -- categorical_acc = {2} -- mae = {3}'.format(epoch, np.mean(mean_tr_loss), np.mean(mean_tr_acc), np.mean(mean_tr_mae)))
        full_model.save('my_second_model_e{0}.h5'.format(epoch))  # creates a HDF5 file 'my_model.h5'
