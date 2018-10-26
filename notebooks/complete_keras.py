import keras
import pickle
import random
import numpy as np
import pandas as pd
import re
import multiprocessing

from tqdm import *
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Input, Reshape, concatenate, Flatten, Activation, LSTM

def generate_input_name(var_name):
    return "input_{0}".format(''.join(c for c in var_name if c.isalnum()))
    
def wrapped__create_learning_dicts_from_trace(p):
    return create_learning_dicts_from_trace(*p)
# reshape X to be [samples, time steps, features]
# How to understand keras feature shape requirements: https://github.com/keras-team/keras/issues/2045
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
ngpus  = len(keras.backend.tensorflow_backend._get_available_gpus())
### CONFIGURATION SETUP END ###
###############################

if __name__ == '__main__':
    # be nice, say hello
    print("Welcome to Felix' master thesis: Deep Learning Next-Activity Prediction Using Subsequence-Enriched Input Data")
    print("Will now train the complete neural network on sequence data, activity data attributes and subsequence info")
    print("Total number of GPUs to be used: {0}".format(ngpus))
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

    # TODO: normalize
    # X = X / float(n_vocab)
    ppool = multiprocessing.Pool(ncores)
    train_traces = []
    traces_for_learning_dicts = [ (t, sp2_col_start_index, n_sp2_features, pfs_col_start_index, n_pfs_features, target_col_start_index, feature_names) for t in traces ]

    with tqdm(total=len(traces), desc="Converting traces to Keras learning data", unit="traces") as pbar:
        for i, _ in tqdm(enumerate(ppool.imap(wrapped__create_learning_dicts_from_trace, traces_for_learning_dicts))):
            pbar.update()
            train_traces.append(_)

    models = []
    model_inputs = []

    # forward all ordinal features
    for ord_var in feature_names[:cat_col_start_index]:
        il = Input(batch_shape=(1,1), name=generate_input_name(ord_var))
        model = Reshape(target_shape=(1,1,))(il)
        model_inputs.append(il)
        models.append(model)

    # create embedding layers for every categorical feature
    for cat_var in categorical_feature_names :
        model = Sequential()
        no_of_unique_cat  = len(feature_dict[cat_var]['to_int'])
        embedding_size = int(min(np.ceil((no_of_unique_cat)/2), 50 ))
        vocab  = no_of_unique_cat+1

        il = Input(batch_shape=(1,1), name=generate_input_name(cat_var))    
        model = Embedding(vocab, embedding_size)(il)
        model = Reshape(target_shape=(1,embedding_size,))(model)

        model_inputs.append(il)
        models.append(model)

    # create input and embedding for sp2/pfs2 features
    learn_sp2 = True
    sequence_embedding = None

    if learn_sp2:
        il = Input(batch_shape=(1,n_sp2_features), name=generate_input_name("sp2"))
        model_inputs.append(il)

        no_of_unique_cat = n_sp2_features
        embedding_size   = int(min(np.ceil((no_of_unique_cat)/2), 50 ))
        vocab  = no_of_unique_cat+1
        sequence_embedding = Embedding(vocab, embedding_size)(il)
        sequence_embedding = Reshape(target_shape=(il.shape[1].value*embedding_size,))(sequence_embedding)
    else:
        # TODO
        pass

    # merge the outputs of the embeddings, and everything that belongs to the most recent activity executions
    main_output = concatenate(models, axis=2)
    main_output = LSTM(25*32, batch_input_shape=(1,), stateful=True)(main_output) # should be multiple of 32 since it trains faster due to np.float32
    # main_output = LSTM(25*32, batch_input_shape=(1,25*32), stateful=True)(main_output) # should be multiple of 32 since it trains faster due to np.float32

    # after LSTM has learned on the sequence, bring in the SP2/PFS features, like in Shibatas paper
    main_output = concatenate([main_output, sequence_embedding])
    main_output = Dense(20*32, activation='relu', name='dense_join')(main_output)
    main_output = Dense(len(feature_dict["concept:name"]["to_int"]), activation='sigmoid', name='dense_final')(main_output)

    full_model = Model(inputs=model_inputs, outputs=[main_output])
    #full_model = multi_gpu_model(full_model, gpus=ngpus)
    full_model.compile(loss='categorical_crossentropy', optimizer='adam')

    print("===> Beginning training....")
    for epoch in range(10):
        mean_tr_acc = []
        mean_tr_loss = []
        for t_idx, t in tqdm(enumerate(train_traces), desc="Epoch {0}/{1}".format(epoch,10)):
            for x,y in zip(t['x'],t['y']):
                tr_loss = full_model.train_on_batch(x, y)
                mean_tr_loss.append(tr_loss)
            full_model.reset_states()

        #print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
        print('Epoch {0} -- training loss = {1}'.format(epoch, np.mean(mean_tr_loss)))
        
    model.save('my_first_model.h5')  # creates a HDF5 file 'my_model.h5'