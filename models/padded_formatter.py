from numpy import percentile
from keras.preprocessing.sequence import pad_sequences
from math import ceil

def find_clean_batch_size(setlen, optimum):
    while setlen % optimum != 0:
        optimum += 1
    return optimum

def format_datasets(model_formatted_data_fn, datapath, target_variable, batch_size):
    train_X, train_Y, test_X, test_Y = model_formatted_data_fn(datapath, target_variable)
    
    # make cutoff step a function of the trace length in each percentile
    mlen = ceil(percentile([len(t) for t in train_Y], 80))

    # remove traces from training which are longer than mlen since they'll be removed anyway
    train_Y = list(filter(lambda t: len(t) <= mlen, train_Y))

    for layer_name in test_X.keys():
        train_X[layer_name] = list(filter(lambda t: len(t) <= mlen, train_X[layer_name]))
    
    # now pad all sequences to same length
    # and reshape into batch format
    batch_size = find_clean_batch_size(len(train_Y), ceil(0.01*len(train_Y)))
    n_y_cols = train_Y[0].shape[1]
    
    train_targets = pad_sequences(train_Y, padding='post').reshape((-1, batch_size, mlen, n_y_cols))
    train_inputs  = {}
    
    for layer_name in test_X.keys():
        n_x_cols = train_X[layer_name][0].shape[1]
        train_inputs[layer_name] = pad_sequences(train_X[layer_name], padding='post').reshape((-1, batch_size, mlen, n_x_cols))
            
    return train_inputs, train_targets, test_X, test_Y