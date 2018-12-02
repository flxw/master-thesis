import pickle
import time
import os.path
import random

from keras.callbacks import Callback

def load_trace_dataset(base_path, purpose='categorical', ttype='test'):
    suffix = "{0}_{1}.pickled".format(purpose, ttype)
    p = os.path.join(base_path, suffix)
    return pickle.load(open(p, "rb"))

def generate_shuffled_bitches(X, Y):
    batches = list(range(len(Y)))
    
    while True:
        random.shuffle(batches)
        for batch_id in batches:
            batch_x = { layer_name: X[layer_name][batch_id] for layer_name in X.keys() }
            batch_y = Y[batch_id]
            
            yield batch_x, batch_y

class StatisticsCallback(Callback):	
    def __init__(self,statistics_df=None, training_batchcount=0, accuracy_metric='accuracy'):	
        self.statistics_df = statistics_df
        self.validation_threshold = training_batchcount - 1
        self.accm = accuracy_metric if accuracy_metric != 'accuracy' else 'acc'	
        self.current_epoch = 0 # early stopping seems to play around with the current epoch
        
    def on_epoch_begin(self,epoch, logs={}):	
        self.training_start = time.time()	
    
    def on_epoch_end(self,epoch, logs={}):
        if epoch != self.current_epoch:
            return
        
        epoch = self.current_epoch
        validation_end = time.time()	
        training_time = self.training_end - self.training_start	
        validation_time = validation_end - self.training_end
        self.statistics_df.values[epoch-1] = [logs['loss'],
                                              logs[self.accm],
                                              logs['val_loss'],
                                              logs['val_' + self.accm],
                                              training_time,
                                              validation_time]
        self.current_epoch += 1
    def on_batch_end(self, batch, logs={}):	
        if batch == self.validation_threshold:	
            self.training_end = time.time() 