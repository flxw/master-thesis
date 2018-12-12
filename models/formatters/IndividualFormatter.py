from numpy import percentile
from keras.preprocessing.sequence import pad_sequences
from math import ceil
from .AbstractFormatter import AbstractFormatter

class IndividualFormatter(AbstractFormatter):
  def format_datasets(model_formatted_data_fn, datapath, target_variable):
    train_X, train_Y, test_X, test_Y = model_formatted_data_fn(datapath, target_variable)

    # reshape into batch format
    batch_size = 1
    n_y_cols = train_Y[0].shape[1]
    train_targets = [ t.reshape((batch_size, -1, n_y_cols)) for t in train_Y ]
    test_targets  = [ t.reshape((batch_size, -1, n_y_cols)) for t in test_Y ]

    train_inputs = {}
    test_inputs  = {}

    for layer_name in test_X.keys():
      n_x_cols = train_X[layer_name][0].shape[1]
      train_inputs[layer_name] = [ t.reshape((batch_size, -1, n_x_cols)) for t in train_X[layer_name] ]
      test_inputs[layer_name]  = [ t.reshape((batch_size, -1, n_x_cols)) for t in test_X[layer_name] ]

    return train_inputs, train_targets, test_inputs, test_targets
