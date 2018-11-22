import numpy  as np
import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Input, Reshape, concatenate, Flatten, Activation, LSTM
from keras.utils  import np_utils
from keras.optimizers import RMSprop
from utils import load_trace_dataset

def prepare_datasets(path_to_original_data, target_variable):
    ### BEGIN DATA LOADING
    train_traces_categorical = load_trace_dataset(path_to_original_data, 'categorical', 'train')
    train_traces_ordinal = load_trace_dataset(path_to_original_data, 'ordinal', 'train')
    train_traces_targets = load_trace_dataset(path_to_original_data, 'target', 'train')
    
    test_traces_categorical = load_trace_dataset(path_to_original_data, 'categorical', 'test')
    test_traces_ordinal = load_trace_dataset(path_to_original_data, 'ordinal', 'test')
    test_traces_targets = load_trace_dataset(path_to_original_data, 'target', 'test')
    
    feature_dict = load_trace_dataset(path_to_original_data, 'mapping', 'dict')
    
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
    train_traces = {'seq_input': [ pd.concat([a,b], axis=1).values for a,b in zip(train_traces_ordinal, train_traces_categorical) ]}
    test_traces  = {'seq_input': [ pd.concat([a,b], axis=1).values for a,b in zip(test_traces_ordinal, test_traces_categorical)   ]}
    
    train_traces_targets = [ t.values for t in train_traces_targets ]
    test_traces_targets  = [ t.values for t in test_traces_targets  ]
    
    return train_traces, train_traces_targets, test_traces, test_traces_targets

def construct_model(n_train_cols, n_target_cols):
    batch_size = None # None translates to unknown batch size
    time_steps = None
    unit_count = n_train_cols + n_target_cols
    
    # [samples, time steps, features]
    il = Input(batch_shape=(batch_size, time_steps, n_train_cols), name='seq_input')

    main_output = LSTM(unit_count,
                       batch_input_shape=(batch_size, time_steps, n_train_cols),
                       stateful=False,
                       return_sequences=True,
                       dropout=0.3)(il)
    main_output = LSTM(unit_count,
                       stateful=False,
                       return_sequences=True,
                       dropout=0.3)(main_output)

    main_output = Dense(n_target_cols, activation='softmax', name='dense_final')(main_output)
    full_model = Model(inputs=[il], outputs=[main_output])
    optimizerator = RMSprop()
    
    full_model.compile(loss='categorical_crossentropy', optimizer=optimizerator, metrics=['accuracy'])
    
    return full_model