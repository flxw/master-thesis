import keras
import pickle
import random
import numpy as np
import pandas as pd
import re

from keras.models import Sequential
from keras.layers import Dense, Embedding, Input

##### CONFIGURATION SETUP ####

data_path = "../logs/bpic2011.xes"
traces_finalpath = data_path.replace(".xes", "_traces_encoded.pickled")
n_sp2_features = 624
n_pfs_features = 25

traces = pickle.load(open(traces_finalpath, "rb"))

### CONFIGURATION SETUP END ###

# shuffle complete traces and create test and training set
random.shuffle(traces)
sep_idx = int(0.8*len(traces))

train_traces = pd.concat(traces[:sep_idx], ignore_index=True)
test_traces  = pd.concat(traces[sep_idx:], ignore_index=True)
assert(sum([len(t) for t in traces]) == len(train_traces)+len(test_traces))

# extract the feature indices
# data is organized like this: normal features | SP2 features | PFS features | TARGET features
trace_columns = train_traces.columns.tolist()
trace_columns = list(map(lambda e: bool(re.match('^TARGET_.+', e)), trace_columns))
target_col_start_index = trace_columns.index(True)
pfs_col_start_index = target_col_start_index - n_pfs_features
sp2_col_start_index  = pfs_col_start_index - n_sp2_features

train_x = train_traces.iloc[:, :sp2_col_start_index].values.astype(np.int16)
test_x  =  test_traces.iloc[:, :sp2_col_start_index].values.astype(np.int16)

# extract only the last column and put each element into an array of its own
train_y = train_traces.iloc[:, target_col_start_index:].values.astype(np.int8)
test_y  =  test_traces.iloc[:, target_col_start_index:].values.astype(np.int8)

model = Sequential()
model.add(Embedding(input_dim=0, input_length=train_x.shape[1], output_dim=300))
#model.add(Dense(train_y.shape[1])) # number of units in last layer is one-hot encoding of output

model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(train_x, train_y, epochs=1, batch_size=130)