from numpy import percentile
from keras.preprocessing.sequence import pad_sequences
from math import ceil

def find_clean_batch_size(setlen, optimum):
    while setlen % optimum != 0:
        optimum += 1
    return optimum

def format_datasets(model_formatted_data_fn, datapath, target_variable, batch_size):
    train_X, train_Y, test_X, test_Y = model_formatted_data_fn(datapath, target_variable)
   
    # reshape into batch format
    batch_size = 1
    n_y_cols = train_Y[0].shape[1]
    train_targets = [ t.reshape((batch_size, -1, n_y_cols)) for t in train_Y ]
    
    train_inputs  = {}
    for layer_name in test_X.keys():
        n_x_cols = train_X[layer_name][0].shape[1]
        train_inputs[layer_name] = [ t.reshape((batch_size, -1, n_x_cols)) for t in train_X[layer_name] ]
            
    return train_inputs, train_targets, test_X, test_Y