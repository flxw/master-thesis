import numpy  as np
import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Input, Reshape, concatenate, Masking, Activation, LSTM
from keras.utils  import np_utils
from keras.optimizers import SGD
from keras.initializers import Zeros
from utils import load_trace_dataset

def prepare_datasets(path_to_original_data, target_variable):    
    train_traces  = load_trace_dataset(path_to_original_data, 'categorical', 'train')
    train_targets = load_trace_dataset(path_to_original_data, 'target', 'train')
    test_traces   = load_trace_dataset(path_to_original_data, 'categorical', 'test')
    test_targets  = load_trace_dataset(path_to_original_data, 'target', 'test')
    feature_dict  = load_trace_dataset(path_to_original_data, 'mapping', 'dict')
    
    train_input  = [ t[target_variable].map(feature_dict[target_variable]['to_int']).values.reshape((-1,1)) for t in train_traces ]
    test_input   = [ t[target_variable].map(feature_dict[target_variable]['to_int']).values.reshape((-1,1)) for t in test_traces ]
    train_targets = np.array([ t.values for t in train_targets ])
    test_targets  = np.array([ t.values for t in test_targets ])
    
    train_input = { 'seq_input': train_input }
    test_input  = { 'seq_input': test_input }
    
    return train_input, train_targets, test_input, test_targets
    
def construct_model(n_train_cols, n_target_cols):
    batch_size  = None # None translates to unknown size
    window_size = None
    
    il = Input(batch_shape=(batch_size,window_size,1), name='seq_input')
    main_output = Masking(mask_value=-1337)(il)
    main_output = Embedding(n_target_cols, 500)(main_output)
    main_output = Reshape(target_shape=(-1,500))(main_output)

    # sizes should be multiple of 32 since it trains faster due to np.float32
    main_output = LSTM(500,
                       batch_input_shape=(batch_size,window_size,1),
                       stateful=False,
                       return_sequences=True,
                       unroll=False,
                       dropout=0.2,
                       kernel_initializer=Zeros())(main_output)
    main_output = LSTM(500,
                       stateful=False,
                       return_sequences=True,
                       unroll=False,
                       dropout=0.2,
                       kernel_initializer=Zeros())(main_output)

    main_output = Dense(n_target_cols, activation='softmax', name='dense_final')(main_output)
    full_model = Model(inputs=[il], outputs=[main_output])
    
    optimizerator = SGD(lr=1)
    full_model.compile(loss='categorical_crossentropy', optimizer=optimizerator, metrics=['categorical_accuracy'])
    
    return full_model
