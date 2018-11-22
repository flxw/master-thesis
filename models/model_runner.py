import argparse
import os
import tqdm
import time
import numpy as np
import pandas as pd
import pickle
import math
import random

# argument setup here
parser = argparse.ArgumentParser(description='The network training script for Felix Wolff\'s master\'s thesis!')
parser.add_argument('model', choices=('evermann', 'schoenig', 'sp2', 'pfs'),
                    help='Which type of model to train.')
parser.add_argument('mode', choices=('padded', 'grouped', 'individual'),
                    help='Which mode to use for feeding the data into the model.')
parser.add_argument('datapath', help='Path of dataset to use for training.')
parser.add_argument('--gpu', default=0, help="CUDA ID of which GPU the model should be placed on to")

args = parser.parse_args()
args.datapath = os.path.abspath(args.datapath)

### CONFIGURATION
es_patience = 20
es_delta = 0
n_epochs = None
remote_path = "/home/felix.wolff2/docker_share"
target_variable = "concept:name"
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
only_categorical = False
### END CONFIGURATION

# load appropriate model
if args.model == 'evermann':
    import evermann_builder as model_builder
    n_epochs = 50
    only_activity = True
elif args.model == 'schoenig':
    import schoenig_builder as model_builder
    n_epochs = 100
elif args.model == 'sp2':
    import sp2_builder as model_builder
    n_epochs = 150
elif args.model == 'pfs':
    import pfs_builder as model_builder
    n_epochs = 150
    
# load appropriate data formatter
if args.mode == 'individual':
    pass
elif args.mode == 'grouped':
    pass
elif args.mode == 'padded':
    import padded_formatter as data_formatter
elif args.mode == 'windowed':
    pass

batch_size = 9
# every model preparation training output will be:
train_X, train_Y, test_X, test_Y = data_formatter.format_datasets(model_builder.prepare_datasets, args.datapath, target_variable, batch_size)

for i in range(0,10):
    print (train_X['seq_input'][i].shape)
    print (train_Y[i].shape)
    print (test_X['seq_input'][i].shape)
    print (test_Y[i].shape)
    print("---")

# train_X, train_Y, test_X, test_Y = data_formatter.reformat_datasets()
# model = model_builder.construct_model()
# statistics = pd.DataFrame(columns=['loss', 'acc', 'val_loss', 'val_acc', 'training_time', 'validation_time'], index=range(0,n_epochs), dtype=np.float32)