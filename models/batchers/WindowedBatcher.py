from .AbstractBatcher import AbstractBatcher
import numpy as np
from math import ceil

def get_windows(trace, k):
  return [ w for w in window_generator(trace, k)]

def window_generator(trace, window_size):
  for i in range(0, len(trace)-window_size):
      yield(trace[i:i+window_size])

class WindowedBatcher(AbstractBatcher):
  window_size = 3

  def format_datasets(model_formatted_data_fn, datapath, target_variable):
    train_X, train_Y, test_X, test_Y = model_formatted_data_fn(datapath, target_variable)
    k = WindowedBatcher.window_size

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
    windowed_test_Y  = np.array([ t[k:,:] for t in test_Y  ])

    layer_name = "seq_input"
    windowed_train_X[layer_name] = np.array([ w for t in train_X[layer_name] for w in get_windows(t, k) ])
    windowed_test_X[layer_name]  = np.array([ np.array([w for w in get_windows(t, k)]) for t in test_X[layer_name]])

    if "sec_input" in train_X.keys():
      layer_name = "sec_input"
      windowed_train_X[layer_name] = np.concatenate([ t[k:,:] for t in train_X[layer_name] ])
      windowed_test_X[layer_name]  = np.array([ t[k:,:] for t in test_X[layer_name] ])

    batch_size = ceil(0.01 * len(windowed_train_Y))
    n_train_batches = int(len(windowed_train_Y) / batch_size)

    for layer_name in test_X.keys():
      n_x_cols = test_X[layer_name][0].shape[1]
      windowed_train_X[layer_name] = np.array_split(windowed_train_X[layer_name], n_train_batches)

    windowed_train_Y = np.array_split(windowed_train_Y, n_train_batches)

    # eliminate empty samples caused by window size
    windowed_test_X = { layer_name: [ t for t in windowed_test_X[layer_name] if t.shape[0] != 0 ] for layer_name in test_X.keys() }
    windowed_test_Y = [ t for t in windowed_test_Y if t.shape[0] != 0]

    print("Windowed batch size:", batch_size)

    return windowed_train_X, windowed_train_Y, windowed_test_X, windowed_test_Y
