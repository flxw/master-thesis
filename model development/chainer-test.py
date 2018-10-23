import chainer
import chainer.functions as F
import chainer.links as L

import pandas as pd
import numpy  as np

import random
import itertools
import pickle
import re

data_path = "../logs/bpic2011.xes"
traces_finalpath = data_path.replace(".xes", "_traces_encoded.pickled")
n_sp2_features = 624
n_pfs_features = 25

xp = np
device_id = -1

traces = pickle.load(open(traces_finalpath, "rb"))

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


train_x = train_traces.iloc[:, :sp2_col_start_index].values.astype(xp.int16)
test_x  =  test_traces.iloc[:, :sp2_col_start_index].values.astype(xp.int16)

# # extract only the last column and put each element into an array of its own
train_y = train_traces.iloc[:, target_col_start_index:].values.astype(xp.int8)
test_y  =  test_traces.iloc[:, target_col_start_index:].values.astype(xp.int8)

train_ds = chainer.datasets.TupleDataset(train_x, train_y)
test_ds  = chainer.datasets.TupleDataset(test_x, test_y)

train_iter = chainer.iterators.SerialIterator(train_ds, 131, repeat=True, shuffle=False)
test_iter  = chainer.iterators.SerialIterator(test_ds,  131, repeat=False, shuffle=False)

gx = gy = None


class SeqDataModel(chainer.Chain):
    def __init__(self, vocab_size, dim_embed=200, dim1=400, dim2=400, dim3=200, class_size=666):
        super(SeqDataModel, self).__init__()
        self.class_size = class_size
        self.vocab_size = vocab_size
        self.dim_embed = dim_embed

        # ss = subsequence
        # sq = sequence
        # co = concatenated
        self.sq_embed1 = L.EmbedID(vocab_size, dim_embed)
        self.sq_lstm2 = L.LSTM(dim_embed, dim1, forget_bias_init=0)
        self.sq_lstm3 = L.LSTM(dim1, dim2, forget_bias_init=0)

        self.co_lin1 = L.Linear(dim2, dim3)
        self.co_lin2 = L.Linear(dim3, vocab_size)

        self.loss_var = chainer.Variable(xp.zeros((), dtype=np.float32))
        self.reset_state()

    def __call__(self, x, train=True):
        global gx, gy
        gx = x
        y = self.sq_embed1(x)
        gy = y
        y = self.sq_lstm2(y)
        y = self.sq_lstm3(y)

        y = self.co_lin1(F.dropout(y, train=train))
        y = F.relu(y)
        y = self.co_lin2(F.dropout(y, train=train))

        return y

    def reset_state(self):
        if self.loss_var is not None:
            self.loss_var.unchain_backward()

        self.loss_var = chainer.Variable(xp.zeros((), dtype=xp.float32))
        self.sq_lstm2.reset_state()
        self.sq_lstm3.reset_state()
        return


gordon = SeqDataModel(vocab_size=train_x.shape[1], class_size=train_y.shape[1])
model = L.Classifier(gordon, accfun=F.accuracy)
optimizer = chainer.optimizers.MomentumSGD().setup(model)

updater = chainer.training.StandardUpdater(train_iter, optimizer, device=device_id)
trainer = chainer.training.Trainer(updater, (1, 'epoch'), out='result')

trainer.extend(chainer.training.extensions.LogReport())
trainer.extend(chainer.training.extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(chainer.training.extensions.Evaluator(test_iter, model, device=device_id))

if device_id != -1:
    model.to_gpu()

trainer.run()