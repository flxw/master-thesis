from numpy import percentile
from keras.preprocessing.sequence import pad_sequences
from math import ceil
import numpy as np

def find_clean_batch_size(setlen, optimum):
    while setlen % optimum != 0:
        optimum += 1
    return optimum

def format_datasets(model_formatted_data_fn, datapath, target_variable, batch_size):
    train_X, train_Y, test_X, test_Y = model_formatted_data_fn(datapath, target_variable)
    
    # loop through every dictionary key and group (since the elements had the same order before, they should have after)
    for input_name in train_X.keys():
        train_X_layer = train_X[input_name]
        grouped_train_X = {}

        # create a dictionary entry for every timeseries length and put the traces in the appropriate bin
        for i in range(0,len(train_X_layer)):
            tl = len(train_X_layer[i])
            elX = np.array(train_X_layer[i])

            if tl in grouped_train_X:
                grouped_train_X[tl].append(elX)
            else:
                grouped_train_X[tl] = [elX]
                
        train_X[input_name] = np.array([np.array(l) for l in grouped_train_X.values()])

    # similarly loop through the targets to cluster them
    grouped_train_Y = {}
    
    for elY in train_Y:
        tl = len(elY)
        
        if tl in grouped_train_Y:
            grouped_train_Y[tl].append(elY)
        else:
            grouped_train_Y[tl] = [elY]
            
    train_Y = np.array([np.array(l) for l in grouped_train_Y.values()])
            
    return train_X, train_Y, test_X, test_Y