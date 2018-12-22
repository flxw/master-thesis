#!/usr/bin/env python3

import argparse
import builders
import batchers
import pkgutil
import importlib

builders = [ modname.replace("builders.","").replace("Builder","").lower()
            for _, modname,_ in pkgutil.walk_packages(path=builders.__path__,
                                                     prefix=builders.__name__+'.',
                                                     onerror=lambda x:None) ]
batchers = [ modname.replace("batchers.","").replace("Batcher","").lower()
            for _, modname,_ in pkgutil.walk_packages(path=batchers.__path__,
                                                     prefix=batchers.__name__+'.',
                                                     onerror=lambda x:None) ]
builders.remove('abstract')
batchers.remove('abstract')

parser = argparse.ArgumentParser(description='PPM-TF | A training framework for predictive process monitoring')
parser.add_argument('builder', choices=builders,
                    help='Which type of model to train.')
parser.add_argument('batcher', choices=batchers,
                    help='Which batching mode to use for feeding the data into the model.')
parser.add_argument('datapath', help='Path of dataset to use for training.')
parser.add_argument('--gpu', default=0, help="CUDA ID of which GPU the model should be placed on to")
parser.add_argument('--output', default='/tmp', help='Target directory to put model and training statistics')

args = parser.parse_args()

# the action begins here
import os
import sys
import time
import numpy as np
import pandas as pd
import pickle
import math
import random
import keras.backend as B
import config

from utils import generate_shuffled_bitches, StatisticsCallback
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback

### CONFIGURATION
args.datapath = os.path.abspath(args.datapath)
args.output = os.path.abspath(args.output)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
### END CONFIGURATION

# load builder and batcher
builder_class = "{0}Builder".format(args.builder.capitalize())
batcher_class = "{0}Batcher".format(args.batcher.capitalize())
builder_importstring = "builders.{0}".format(builder_class)
batcher_importstring = "batchers.{0}".format(batcher_class)

model_builder  = getattr(importlib.import_module(builder_importstring), builder_class)
data_formatter = getattr(importlib.import_module(batcher_importstring), batcher_class)

# every model preparation training output will be:
# 3D for every train_X / train_Y element
# 2D for every test_X / test_Y element
train_X, train_Y, test_X, test_Y = data_formatter.format_datasets(model_builder.prepare_datasets,
                                                                  args.datapath, config.target_variable)
n_X_cols = [test_X[name][0].shape[-1] for name in test_X.keys()]
n_Y_cols = test_Y[0].shape[-1]
train_batchcount = len(train_Y)
test_batchcount = len(test_Y)

model = model_builder.construct_model(n_X_cols, n_Y_cols, args.batcher == 'windowed')
statistics_df = pd.DataFrame(columns=['loss', 'acc', 'val_loss', 'val_acc', 'training_time', 'validation_time'],
                             index=range(0,model_builder.n_epochs), dtype=np.float32)

cb_earlystopper = EarlyStopping(monitor='val_loss',
                      min_delta=config.es_delta,
                      patience=config.es_patience,
                      verbose=1,
                      mode='auto',
                      restore_best_weights=False)

cb_checkpointer = ModelCheckpoint(monitor='val_loss',
                                  filepath="{0}/{1}_{2}.hdf5".format(args.output, args.builder, args.batcher),
                                  verbose=1,
                                  save_best_only=True)

cb_statistics = StatisticsCallback(statistics_df=statistics_df,
                                   training_batchcount = train_batchcount,
                                   accuracy_metric=model.metrics[0])

cbs = [cb_earlystopper, cb_checkpointer, cb_statistics]

if args.builder == 'evermann': # why? See Implementation in evermann2016
    cb = LambdaCallback(on_epoch_begin=lambda epoch, logs: B.set_value(model.optimizer.decay, .75) if epoch==24 else False)
    cbs.append(cb)

model.fit_generator(generate_shuffled_bitches(train_X, train_Y),
                    steps_per_epoch=train_batchcount,
                    epochs=model_builder.n_epochs,
                    callbacks=cbs,
                    use_multiprocessing=False,
                    validation_data=generate_shuffled_bitches(test_X, test_Y),
                    validation_steps=test_batchcount,
                    shuffle=False)

print("Overall validation accuracy: ", statistics_df['val_acc'].max(axis=0))
statistics_df.dropna(axis=0, how='all', inplace=True)
statistics_df = statistics_df[:-1]
statistics_df.to_pickle("{0}/{1}_{2}_stats.pickled".format(args.output, args.builder, args.batcher))
