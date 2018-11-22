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
    import formatters.individual as data_formatter
elif args.mode == 'grouped':
    import formatters.grouped as data_formatter
elif args.mode == 'padded':
    import formatters.padded as data_formatter
elif args.mode == 'windowed':
    pass

# every model preparation training output will be:
# 3D for every train_X / train_Y element
# 2D for every test_X / test_Y element
train_X, train_Y, test_X, test_Y = data_formatter.format_datasets(model_builder.prepare_datasets, args.datapath, target_variable)
n_X_cols = test_X['seq_input'][0].shape[1]
n_Y_cols = test_Y[0].shape[1]

model = model_builder.construct_model(n_X_cols, n_Y_cols)
statistics_df = pd.DataFrame(columns=['loss', 'acc', 'val_loss', 'val_acc', 'training_time', 'validation_time'], index=range(0,n_epochs), dtype=np.float32)

# now that the data is ready, the model can be trained!
last_tr_acc   = 0
last_tr_loss  = 0
last_val_acc  = 0
last_val_loss = 0
best_val_loss = math.inf
current_patience = es_patience

epoch_iterator = tqdm.trange(1, n_epochs+1)
for epoch in epoch_iterator:
    tr_accs = []
    tr_losses = []
    val_accs = []
    val_losses = []
    
    # training an epoch
    t_start = time.time()
    
    # shuffle batches for every epoch
    batches = list(range(len(train_Y)))
    random.shuffle(batches)
    
    # train the network on a batch
    for batch_id in tqdm.tqdm(batches,
                       desc="acc: {0:.2f} | loss: {1:.2f} | val_acc {2:.2f} | val_loss: {3:.2f}".format(last_tr_acc, last_tr_loss, last_val_acc, last_val_loss)):
        # Each first-level element is a batch
        batch_x = { layer_name: train_X[layer_name][batch_id] for layer_name in train_X.keys() }
        batch_y = train_Y[batch_id]
        
        l,a = model.train_on_batch(batch_x, batch_y)
        tr_losses.append(l)
        tr_accs.append(a)

    last_tr_acc = np.mean(tr_accs)
    last_tr_loss = np.mean(tr_losses)
    training_time = time.time() - t_start
    
    # validating the epoch result
    t_start = time.time()
    for batch_id in range(len(test_Y)):
        batch_y = test_Y[batch_id].reshape((1,-1,n_Y_cols))
        batch_x = { layer_name: np.array([test_X[layer_name][batch_id]]) for layer_name in test_X.keys() }
        
        l,a = model.test_on_batch(x=batch_x, y=batch_y)
        val_losses.append(l)
        val_accs.append(a)

    last_val_acc = np.mean(val_accs)
    last_val_loss = np.mean(val_losses)
    validation_time = time.time() - t_start
    
    statistics_df.values[epoch-1] = [last_tr_loss,
                                  last_tr_acc,
                                  last_val_loss,
                                  last_val_acc,
                                  training_time,
                                  validation_time]
    
    if best_val_loss > last_val_loss:
        tqdm.tqdm.write("Decreased loss from {0:.2f} to {1:.2f} - saving model!".format(best_val_loss, last_val_loss))
        best_val_loss = last_val_loss
        current_patience = es_patience
        model.save("{0}/{1}/{2}/best_val_loss_e{3}.hdf5".format(remote_path, args.model, args.mode, epoch))
    else:
        current_patience -= 1

    if current_patience == 0:
        tqdm.tqdm.write("Early stopping, since loss has not improved for {0} epochs".format(es_patience))
        model.save("{0}/{1}/{2}/best_val_loss_e{3}.hdf5".format(remote_path, args.model, args.mode, epoch))
        epoch_iterator.close()
        break

statistics_df.to_pickle("{0}/{1}/{2}/train_statistics.pickled".format(remote_path, args.model, args.mode))
