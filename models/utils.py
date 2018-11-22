import pickle
import time
import os.path

from keras.callbacks import Callback

def load_trace_dataset(base_path, purpose='categorical', ttype='test'):
    suffix = "{0}_{1}.pickled".format(purpose, ttype)
    p = os.path.join(base_path, suffix)
    return pickle.load(open(p, "rb"))