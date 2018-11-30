import numpy as np
from keras.preprocessing.sequence import pad_sequences
from math import ceil

k = 2
batch_size = 32

def find_clean_batch_size(setlen, optimum):
    while setlen % optimum != 0:
        optimum += 1
    return optimum

def get_windows(trace):
    return [ w for w in window_generator(trace, k)]

def window_generator(trace, window_size):
    for i in range(0, len(trace)-window_size):
        yield(trace[i:i+window_size])

def format_datasets(model_formatted_data_fn, datapath, target_variable):
    train_X, train_Y, test_X, test_Y = model_formatted_data_fn(datapath, target_variable)
    
    assert(len(train_X['seq_input'][0].shape) == 2)
    assert(len(train_Y[0].shape) == 2)
    assert(len(test_X['seq_input'][0].shape) == 2)
    assert(len(test_Y[0].shape) == 2)
   
    # reshape into batch format
    windowed_train_X = {}
    windowed_test_X = {}
    
    # at this point, train_Y is an array of 2D numpy arrays
    n_y_cols = train_Y[0].shape[1]
    windowed_train_Y = np.concatenate([ t[k:,:] for t in train_Y ])
    windowed_test_Y  = np.concatenate([ t[k:,:] for t in test_Y  ])
    
    for layer_name in test_X.keys():
        windowed_train_X[layer_name] = np.array([ w for t in train_X[layer_name] for w in get_windows(t) ])
        windowed_test_X[layer_name]  = np.array([ w for t in test_X[layer_name]  for w in get_windows(t) ])
        
    train_bs = find_clean_batch_size(windowed_train_X['seq_input'].shape[0], batch_size)
    test_bs  = find_clean_batch_size(windowed_test_X['seq_input'].shape[0], batch_size)

    for layer_name in test_X.keys():
        n_x_cols = test_X[layer_name][0].shape[1]
        windowed_train_X[layer_name] = windowed_train_X[layer_name].reshape((-1, train_bs, k, n_x_cols))
        windowed_test_X[layer_name]  = windowed_test_X[layer_name].reshape((-1, test_bs, k, n_x_cols))
        
    # find find better batch size here for 
    windowed_train_Y = windowed_train_Y.reshape((-1, train_bs, n_y_cols))
    windowed_test_Y  =  windowed_test_Y.reshape((-1, test_bs, n_y_cols))
    print("Using batch size", train_bs)
    
    return windowed_train_X, windowed_train_Y, windowed_test_X, windowed_test_Y