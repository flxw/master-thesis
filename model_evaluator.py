#!/usr/bin/env python3
import sys
sys.path.append('./models')

import keras
import pickle
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from matplotlib.ticker import MaxNLocator
from utils import load_trace_dataset

from evaluation_helpers import *
from batchers.IndividualBatcher import IndividualBatcher
from batchers.WindowedBatcher import WindowedBatcher
from builders.EvermannBuilder import EvermannBuilder
from builders.Sp2Builder import Sp2Builder
from builders.PfsBuilder import PfsBuilder
from tqdm import tqdm_notebook

##### CONFIGURATION SETUP ####
log_name = sys.argv[1]
datapath = "./logs/{0}".format(log_name)
percentiles = 10 if log_name == 'helpdesk' else 20
target_variable = "concept:name"
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
### CONFIGURATION SETUP END ###

test_traces_categorical = load_trace_dataset(datapath, 'categorical', 'test')
test_traces_ordinal = load_trace_dataset(datapath, 'ordinal', 'test')
test_traces_sp2 = load_trace_dataset(datapath, 'sp2', 'test')
test_traces_pfs = load_trace_dataset(datapath, 'pfs', 'test')
test_targets = load_trace_dataset(datapath, 'target', 'test')
feature_dict = load_trace_dataset(datapath, 'mapping', 'dict')

_, _, evm_ind_X, evm_ind_Y = IndividualBatcher.format_datasets(EvermannBuilder.prepare_datasets,
                                                           datapath, target_variable)
_, _, pfs_ind_X, pfs_ind_Y = IndividualBatcher.format_datasets(PfsBuilder.prepare_datasets,
                                                           datapath, target_variable)
_, _, sp2_ind_X, sp2_ind_Y = IndividualBatcher.format_datasets(Sp2Builder.prepare_datasets,
                                                        datapath, target_variable)

_, _, evm_win_X, evm_win_Y = WindowedBatcher.format_datasets(EvermannBuilder.prepare_datasets,
                                                           datapath, target_variable)
_, _, pfs_win_X, pfs_win_Y = WindowedBatcher.format_datasets(PfsBuilder.prepare_datasets,
                                                      datapath, target_variable)
_, _, sp2_win_X, sp2_win_Y = WindowedBatcher.format_datasets(Sp2Builder.prepare_datasets,
                                                        datapath, target_variable)

# one trace per batch
evm_ind_model = keras.models.load_model("../docker_share/{0}/evermann_individual.hdf5".format(log_name))
#sch_ind_model = keras.models.load_model("../docker_share/{0}/schoenig_individual.hdf5".format(log_name))
#sp2_ind_model = keras.models.load_model("../docker_share/{0}/sp2_individual.hdf5".format(log_name))
#pfs_ind_model = keras.models.load_model("../docker_share/{0}/pfs_individual.hdf5".format(log_name))

evm_ind_precisions = calculate_percentile_precisions(evm_ind_model, evm_ind_X, evm_ind_Y, percentiles)
#sch_ind_precisions = calculate_percentile_precisions(sch_ind_model, sp2_ind_X, sp2_ind_Y, percentiles)
#sp2_ind_precisions = calculate_percentile_precisions(sp2_ind_model, sp2_ind_X, sp2_ind_Y, percentiles)
#pfs_ind_precisions = calculate_percentile_precisions(pfs_ind_model, pfs_ind_X, pfs_ind_Y, percentiles)

pickle.dump(evm_ind_precisions, open("../docker_share/{0}/evermann_individual_evaluation.pickled".format(log_name), "wb"))
#pickle.dump(sch_ind_precisions, open("../docker_share/{0}/schoenig_individual_evaluation.pickled".format(log_name), "wb"))
#pickle.dump(sp2_ind_precisions, open("../docker_share/{0}/sp2_individual_evaluation.pickled".format(log_name), "wb"))
#pickle.dump(pfs_ind_precisions, open("../docker_share/{0}/pfs_individual_evaluation.pickled".format(log_name), "wb"))

# grouped batches
evm_grp_model = keras.models.load_model("../docker_share/{0}/evermann_grouped.hdf5".format(log_name))
sch_grp_model = keras.models.load_model("../docker_share/{0}/schoenig_grouped.hdf5".format(log_name))
sp2_grp_model = keras.models.load_model("../docker_share/{0}/sp2_grouped.hdf5".format(log_name))
pfs_grp_model = keras.models.load_model("../docker_share/{0}/pfs_grouped.hdf5".format(log_name))

evm_grp_precisions = calculate_percentile_precisions(evm_grp_model, evm_ind_X, evm_ind_Y, percentiles)
sch_grp_precisions = calculate_percentile_precisions(sch_grp_model, sp2_ind_X, sp2_ind_Y, percentiles)
sp2_grp_precisions = calculate_percentile_precisions(sp2_grp_model, sp2_ind_X, sp2_ind_Y, percentiles)
pfs_grp_precisions = calculate_percentile_precisions(pfs_grp_model, pfs_ind_X, pfs_ind_Y, percentiles)

pickle.dump(evm_grp_precisions, open("../docker_share/{0}/evermann_grouped_evaluation.pickled".format(log_name), "wb"))
pickle.dump(sch_grp_precisions, open("../docker_share/{0}/schoenig_grouped_evaluation.pickled".format(log_name), "wb"))
pickle.dump(sp2_grp_precisions, open("../docker_share/{0}/sp2_grouped_evaluation.pickled".format(log_name), "wb"))
pickle.dump(pfs_grp_precisions, open("../docker_share/{0}/pfs_grouped_evaluation.pickled".format(log_name), "wb"))

# padded batches
evm_pad_model = keras.models.load_model("/home/felix.wolff2/docker_share/{0}/evermann_padded.hdf5".format(log_name))
sch_pad_model = keras.models.load_model("/home/felix.wolff2/docker_share/{0}/schoenig_padded.hdf5".format(log_name))
sp2_pad_model = keras.models.load_model("/home/felix.wolff2/docker_share/{0}/sp2_padded.hdf5".format(log_name))
pfs_pad_model = keras.models.load_model("/home/felix.wolff2/docker_share/{0}/pfs_padded.hdf5".format(log_name))

evm_pad_precisions = calculate_percentile_precisions(evm_pad_model, evm_ind_X, evm_ind_Y, percentiles)
sch_pad_precisions = calculate_percentile_precisions(sch_pad_model, sp2_ind_X, sp2_ind_Y, percentiles)
sp2_pad_precisions = calculate_percentile_precisions(sp2_pad_model, sp2_ind_X, sp2_ind_Y, percentiles)
pfs_pad_precisions  = calculate_percentile_precisions(pfs_pad_model, pfs_ind_X, pfs_ind_Y, percentiles)

pickle.dump(evm_pad_precisions, open("../docker_share/{0}/evermann_padded_evaluation.pickled".format(log_name), "wb"))
pickle.dump(sch_pad_precisions, open("../docker_share/{0}/schoenig_padded_evaluation.pickled".format(log_name), "wb"))
pickle.dump(sp2_pad_precisions, open("../docker_share/{0}/sp2_padded_evaluation.pickled".format(log_name), "wb"))
pickle.dump(pfs_pad_precisions, open("../docker_share/{0}/pfs_padded_evaluation.pickled".format(log_name), "wb"))

# windowed batches
evm_win_model = keras.models.load_model("../docker_share/{0}/evermann_windowed.hdf5".format(log_name))
sch_win_model = keras.models.load_model("../docker_share/{0}/schoenig_windowed.hdf5".format(log_name))
sp2_win_model = keras.models.load_model("../docker_share/{0}/sp2_windowed.hdf5".format(log_name))
pfs_win_model = keras.models.load_model("../docker_share/{0}/pfs_windowed.hdf5".format(log_name))

window_size = 3
evm_win_precisions = calculate_windowed_precision(evm_win_model, evm_win_X, evm_win_Y, window_size, percentiles)
sch_win_precisions = calculate_windowed_precision(sch_win_model, sp2_win_X, sp2_win_Y, window_size, percentiles)
sp2_win_precisions = calculate_windowed_precision(sp2_win_model, sp2_win_X, sp2_win_Y, window_size, percentiles)
pfs_win_precisions = calculate_windowed_precision(pfs_win_model, pfs_win_X, pfs_win_Y, window_size, percentiles)

pickle.dump(evm_win_precisions, open("../docker_share/{0}/evermann_windowed_evaluation.pickled".format(log_name), "wb"))
pickle.dump(sch_win_precisions, open("../docker_share/{0}/schoenig_windowed_evaluation.pickled".format(log_name), "wb"))
pickle.dump(sp2_win_precisions, open("../docker_share/{0}/sp2_windowed_evaluation.pickled".format(log_name), "wb"))
pickle.dump(pfs_win_precisions, open("../docker_share/{0}/pfs_windowed_evaluation.pickled".format(log_name), "wb"))
