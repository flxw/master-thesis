#!/usr/bin/env python3

import pickle
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker as mtick
import plot_styles
import os

def plot_precisions(evm_prec, sch_prec, sp2_prec, pfs_prec):
    percentiles = len(evm_prec)
    percentile_steps = int(100/percentiles)
    xrange = list(range(percentile_steps,(1+percentiles)*percentile_steps,percentile_steps))
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    d = {
        'EVM': evm_prec,
        'SCH': sch_prec,
        'SP2': sp2_prec,
        'PFS': pfs_prec
    }
    
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter(fmt))
    plt.xticks(xrange)
    plt.yticks(np.arange(0,1,.1))
    ax.set_ylim([0,1])
    d = pd.DataFrame(d, index=xrange)

    ax = sns.lineplot(data=d, ax=ax)
    ax.set(xlabel='Progress toward completion', ylabel='Accuracy')
    ax.legend(loc='upper right')
    
def plot_statistics(evm_stats, sch_stats, sp2_stats, pfs_stats):
    cols = ['loss', 'val_loss','acc', 'val_acc']
    secy_cols = ['acc', 'val_acc']
    plotstyle = ['-','-','-.','-.']
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(30,15), sharex=True)
    axs = axs.reshape((-1))

    for i in range(3):
        axs[i].set(xlabel='Epochs', ylabel='Loss')

    evm_stats[cols].plot(secondary_y=secy_cols, kind='line', style=plotstyle, legend=False, ax=axs[0]).set_title('EVM')
    sch_stats[cols].plot(secondary_y=secy_cols, kind='line', style=plotstyle, legend=False, ax=axs[1]).set_title('SCH')
    sp2_stats[cols].plot(secondary_y=secy_cols, kind='line', style=plotstyle, legend=False, ax=axs[2]).set_title('SP2')
    pfs_stats[cols].plot(secondary_y=secy_cols, kind='line', style=plotstyle, legend=False, ax=axs[3]).set_title('PFS')
    
    handles,labels = [],[]
    for ax in [fig.axes[0], fig.axes[4]]:
        for h,l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
                            
    plt.legend(handles=handles,
               loc="upper left",
               bbox_to_anchor=(0, 1.2),
               ncol=4)
    
def plot_heatmap(dataset_name):
    df = pd.read_pickle("/home/felix.wolff2/docker_share/{0}/correlation_crosstab.pickled".format(dataset_name))
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(df, mask=mask, vmin=0, vmax=1, cmap="OrRd", annot=True)
    
datasets = ['bpic2011', 'bpic2012', 'bpic2015_1', 'bpic2015_2', 'bpic2015_3', 'bpic2015_4', 'bpic2015_5', 'helpdesk']

models = ['evermann', 'schoenig', 'sp2', 'pfs']
batching = ['individual', 'grouped', 'padded', 'windowed']
name_map = {'individual':'Individual',
            'grouped':'Grouping',
            'padded':'Padding',
            'windowed':'Windowing'}

for d in datasets:
    try:
        os.mkdir("{0}".format(d))
    except:
        pass
    
    acc_df = pd.DataFrame(columns=['Batcher', 'Model', 'Validation accuracy'])
    time_df = pd.DataFrame(columns=['Batcher', 'Model', 'Time'])
        
    for b in batching:
        # Create percentile plot
        evm_df = pickle.load(open("/home/felix.wolff2/docker_share/{0}/evermann_{1}_evaluation.pickled".format(d,b), "rb"))
        sch_df = pickle.load(open("/home/felix.wolff2/docker_share/{0}/schoenig_{1}_evaluation.pickled".format(d,b), "rb"))
        sp2_df = pickle.load(open("/home/felix.wolff2/docker_share/{0}/sp2_{1}_evaluation.pickled".format(d,b), "rb"))
        pfs_df = pickle.load(open("/home/felix.wolff2/docker_share/{0}/pfs_{1}_evaluation.pickled".format(d,b), "rb"))
        
        plot_precisions(evm_df, sch_df, sp2_df, pfs_df)
                        
        plt.tight_layout()
        plt.savefig("{0}/{1}_stability.pdf".format(d,b))
        plt.close()
        
        # Create performance curve plot
        evm_stats = pd.read_pickle('/home/felix.wolff2/docker_share/{0}/evermann_{1}_stats.pickled'.format(d,b))
        sch_stats = pd.read_pickle('/home/felix.wolff2/docker_share/{0}/schoenig_{1}_stats.pickled'.format(d,b))
        sp2_stats = pd.read_pickle('/home/felix.wolff2/docker_share/{0}/sp2_{1}_stats.pickled'.format(d,b))
        pfs_stats = pd.read_pickle('/home/felix.wolff2/docker_share/{0}/pfs_{1}_stats.pickled'.format(d,b))
        
        plot_statistics(evm_stats, sch_stats, sp2_stats, pfs_stats)
        plt.tight_layout()
        plt.savefig("{0}/{1}_loss_acc_curve.pdf".format(d,b))
        plt.close()
        
        # Grab maximum accuracies here
        acc_df = acc_df.append({'Batcher': name_map[b], 'Model': 'EVM', 'Validation accuracy': evm_stats['val_acc'].max()}, ignore_index=True)
        acc_df = acc_df.append({'Batcher': name_map[b], 'Model': 'SCH', 'Validation accuracy': sch_stats['val_acc'].max()}, ignore_index=True)
        acc_df = acc_df.append({'Batcher': name_map[b], 'Model': 'SP2', 'Validation accuracy': sp2_stats['val_acc'].max()}, ignore_index=True)
        acc_df = acc_df.append({'Batcher': name_map[b], 'Model': 'PFS', 'Validation accuracy': pfs_stats['val_acc'].max()}, ignore_index=True)
        
        # Grab training times here
        time_df = time_df.append({'Batcher': name_map[b], 'Model': 'EVM', 'Time': evm_stats['training_time'].mean()}, ignore_index=True)
        time_df = time_df.append({'Batcher': name_map[b], 'Model': 'SCH', 'Time': sch_stats['training_time'].mean()}, ignore_index=True)
        time_df = time_df.append({'Batcher': name_map[b], 'Model': 'SP2', 'Time': sp2_stats['training_time'].mean()}, ignore_index=True)
        time_df = time_df.append({'Batcher': name_map[b], 'Model': 'PFS', 'Time': pfs_stats['training_time'].mean()}, ignore_index=True)
        
    # Plot accuracies
    plt.yticks(np.arange(0,1,.1))
    plotax = sns.barplot(data=acc_df,
                         x="Batcher",
                         y="Validation accuracy",
                         hue="Model")
    plotax.set(xlabel='Batch formatting strategy', ylabel='Accuracy', ylim=[0,1])
    plotax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("{0}/accuracies.pdf".format(d))
    plt.close()
    
    # plot times
    plotax = sns.barplot(data=time_df,
                         x="Batcher",
                         y="Time",
                         hue="Model")
    plotax.set(xlabel='Batch formatting strategy', ylabel='Time [s]')
    plotax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("{0}/train_timings.pdf".format(d))
    plt.close()
    
    if d != 'helpdesk':
      # plot heatmap
      plot_heatmap(d)
      plt.tight_layout()
      plt.savefig("{0}/correlation-heatmap.pdf".format(d))
      plt.close()
      
