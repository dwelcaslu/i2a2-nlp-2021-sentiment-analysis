import itertools
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, balanced_accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns


def performance_measures(y_true, y_pred, class_names=None):
    classes_in_eval = list(set(y_true).union(set(y_pred)))
    if class_names is None:
        class_names = [str(i) for i in range(len(classes_in_eval))]
    classes_labels = [i for i in range(len(class_names))]

    results = {'accuracy': format_measure(accuracy_score(y_true, y_pred)),
               'balanced_accuracy': format_measure(balanced_accuracy_score(y_true, y_pred)),
               'balanced_accuracy adjusted': format_measure(balanced_accuracy_score(y_true, y_pred, adjusted=True)),
               'precision': format_measure(precision_score(y_true, y_pred, average=None)),
               'recall': format_measure(recall_score(y_true, y_pred, average=None)),
               'f1_score': format_measure(f1_score(y_true, y_pred, average=None))
              }
    print("Classes evaluated:", [class_names[i] for i in classes_in_eval])
    print('Correctly classified......: ' + str(accuracy_score(y_true, y_pred, normalize=False)) + '/' + str(len(y_pred)))
    print('Accuracy (simple) ........: ' + str(results['accuracy']))
    print('Balanced acc.  ...........: ' + str(results['balanced_accuracy']))
    print('Balanced acc. (adjusted)..: ' + str(results['balanced_accuracy adjusted']))
    print('Acc. norm (Recall avgs) ..: ' + str(format_measure(results['recall'].mean())))
    print('Precision = tp / (tp + fp): ' + str(results['precision']) + ' -> accuracy of positive predictions')
    print('Recall    = tp / (tp + fn): ' + str(results['recall']) + ' -> sensibility (true positive rate)')
    print('F1 score .................: ' + str(results['f1_score']) + ' -> harmonic mean')
    plt.figure(figsize=(14, 8))
    plt.subplot(1, 2, 1)
    plot_confusion_matrix(confusion_matrix(y_true, y_pred, labels=classes_labels), classes=class_names)
    plt.subplot(1, 2, 2)
    plot_confusion_matrix(confusion_matrix(y_true, y_pred, labels=classes_labels), classes=class_names, normalize=True)
    plt.tight_layout(w_pad=10.0)
    plt.show()

    return results

def format_measure(measure):
    if isinstance(measure, float):
        return format(round(measure, 2), '.2f')
    else:
        return np.around(measure.astype(float), decimals=2)

def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title  = 'Confusion Matrix (%)'
    else:
        title  = 'Confusion Matrix (Quantity)'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=80)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = (cm.min() + cm.max()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')


def plot_classification_report(report, save_fig=False, fig_name='classification_report'):
    plot_chart(report)
    if save_fig:
        plt.savefig(f'{fig_name}.png', dpi=100, format='png', bbox_inches='tight')
    plt.show()

def plot_chart(classification_report, title='Classification report ', cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857
    '''
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2:]:
        if line == '':
            break
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        # print(v)
        plotMat.append(v)

    # print('plotMat: {0}'.format(plotMat))
    # print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)

def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap, vmin=0.0, vmax=1.0)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))

def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857
    By HYRY
    '''

    pc.update_scalarmappable()
    ax = pc.axes
    #ax = pc.axes# FOR LATEST MATPLOTLIB
    #Use zip BELOW IN PYTHON 3
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def plot_model_hist(historic, fig1_metric='loss', fig2_metric='acc', title='', figsize=(15, 5)):

    plt.figure(figsize=figsize)

    plt.subplot(1,2,1)
    plt.title(f'{title} {fig1_metric}')
    plt.plot(historic[fig1_metric], label=f'train {fig1_metric}')
    plt.plot(historic[f'val_{fig1_metric}'], label=f'valid. {fig1_metric}')
    if fig1_metric in ['acc', 'f1_score', 'recall', 'precision']:
        plt.ylim([0, 1])
    plt.grid()
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel(fig1_metric)

    plt.subplot(1,2,2)
    plt.title(f'{title} {fig2_metric}')
    plt.plot(historic[fig2_metric], label=f'train {fig2_metric}')
    plt.plot(historic[f'val_{fig2_metric}'], label=f'valid. {fig2_metric}')
    if fig2_metric in ['acc', 'f1_score', 'recall', 'precision']:
        plt.ylim([0, 1])
    plt.grid()
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel(fig2_metric)

    plt.show()


def plot_corr_table(x_array, features_names=None, figsize=(10, 8), vmin_max=None, title='', xy_labels=('', ''), n_decimals=2):
    if vmin_max is None:
        vmin_max = (x_array.min(), x_array.max())
    if features_names is None:
        features_names = [i+1 for i in range(x_array.shape[0])]
    # plot the heatmap:
    plt.figure(figsize=figsize)
    plt.title(title)
    sns.heatmap(x_array, xticklabels=features_names, cmap='RdBu_r', annot=np.round(x_array, n_decimals),
                yticklabels=features_names, vmin=vmin_max[0], vmax=vmin_max[1])
    plt.xlabel(xy_labels[0])
    plt.ylabel(xy_labels[1])
    plt.show()


def plot_embeddings(x_array, y_array):
    # Calculating the embeddings:
    pca = PCA(n_components=2)
    x_array_pca = pca.fit_transform(x_array)
    tsne = TSNE(n_components=2)
    x_array_tsne = tsne.fit_transform(x_array)
    
    
    plt.figure(figsize=(18, 8))
    plt.subplot(1,2,1)
    plt.title(f'PCA - explained_variance_ratio: {round(sum(pca.explained_variance_ratio_), 4)}')
    for val in np.unique(y_array):
        x_values = x_array_pca[np.where(y_array == val)]
        plt.scatter(x_values[:, 0], x_values[:, 1], label=val, alpha=0.25)
    plt.legend(loc='best')
    plt.subplot(1,2,2)
    plt.title(f'TSNE - kl_divergence: {round(tsne.kl_divergence_, 4)}')
    for val in np.unique(y_array):
        x_values = x_array_tsne[np.where(y_array == val)]
        plt.scatter(x_values[:, 0], x_values[:, 1], label=val, alpha=0.25)
    plt.legend(loc='best')
    plt.show()
