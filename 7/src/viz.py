import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(history):
    metric_list = [m for m in list(history.keys()) if m != 'lr']
    size = (len(metric_list)//2)  # adjust vertical space to fit all metrics
    fig, axes = plt.subplots(size+1, 1, sharex='col', figsize=(20, (size+1) * 4))
    if size > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for index in range(len(metric_list)//2):
        # assuming metric_list has pairs of train-val ['loss', 'acc', 'val_loss', 'val_acc']
        metric_name = metric_list[index]
        val_metric_name = metric_list[index+size]
        axes[index].plot(history[metric_name], label='Train %s' % metric_name)
        axes[index].plot(history[val_metric_name], label='Validation %s' % metric_name)
        axes[index].legend(loc='best', fontsize=16)
        axes[index].set_title(metric_name)
        if 'loss' in metric_name:
            axes[index].axvline(np.argmin(history[metric_name]), linestyle='dashed')
            axes[index].axvline(np.argmin(history[val_metric_name]), linestyle='dashed', color='orange')
        else:
            axes[index].axvline(np.argmax(history[metric_name]), linestyle='dashed')
            axes[index].axvline(np.argmax(history[val_metric_name]), linestyle='dashed', color='orange')
    
    axes[-1].plot(history['lr'], label='LR')
    axes[-1].set_title('LR')

    for ax in axes:
        ax.set_xlim([0,len(history['lr'])])
        ax.grid(axis='y')

    plt.xlabel('Epochs', fontsize=16)
    plt.show()