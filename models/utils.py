import pickle
import time
import os.path

from keras.callbacks import Callback

def load_trace_dataset(base_path, purpose='categorical', ttype='test'):
    suffix = "{0}_{1}.pickled".format(purpose, ttype)
    p = os.path.join(base_path, suffix)
    return pickle.load(open(p, "rb"))

class StatisticsCallback(Callback):
    def __init__(self,training_batchcount,statistics_df=None, accuracy_metric='accuracy'):
        self.statistics_df = statistics_df
        self.validation_threshold = training_batchcount - 1
        self.accm = accuracy_metric if accuracy_metric != 'accuracy' else 'acc'

    def on_epoch_begin(self,epoch, logs={}):
        self.training_start = time.time()

    def on_epoch_end(self,epoch, logs={}):
        validation_end = time.time()
        training_time = self.training_end - self.training_start
        validation_time = validation_end - self.training_end
        self.statistics_df.values[epoch-1] = [logs['loss'], logs[self.accm], logs['val_loss'], logs['val_' + self.accm] ,training_time, validation_time]
        
    def on_batch_end(self,batch,logs={}):
        if batch == self.validation_threshold:
            self.training_end = time.time()