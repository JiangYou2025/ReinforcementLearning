# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 13:58:22 2019

@author: you
"""

from textwrap import wrap
import re
import itertools 
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
from sklearn.metrics import confusion_matrix

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
"""
vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

harvest = np.random.random([15,17])

mat, x, y = harvest, vegetables, farmers
f = plot(harvest, vegetables, farmers)
color='Reds'

fig = sns.heatmap(mat, robust=True, annot=True, fmt='.2f', cmap='RdYlGn', center =0, xticklabels=x, yticklabels=y )

fig= matplotlib.figure.Figure(figsize=(14,8))
fig, ax = plt.subplots(figsize=(10,8))         # Sample figsize in inches
sns.heatmap(mat,annot=True, fmt='.2f', cmap='RdYlGn', center =0, xticklabels=x, yticklabels=y )
fig
"""

class HeatMap():
    def __init__(self):
        self.idx = 0
        
#fig, ax = plt.subplots(figsize=(14,6))    
    def get_heat_map(self, mat, x, y, color):
        self.idx+=1
        fig, ax = plt.subplots(figsize=(14,6), num=self.idx%2, clear=True)
  
        if color == "RdYlGn":     # Sample figsize in inches
            sns.heatmap(mat,annot=True, fmt='.2f', cmap=color, center = 0, xticklabels=y, yticklabels=x, ax=ax) 
        else:
            sns.heatmap(mat,annot=True, fmt='.2f', cmap=color, vmin = 0, xticklabels=y, yticklabels=x, ax=ax)
        
        plt.close() 
        return fig
"""

i=0

#fig, ax = plt.subplots(figsize=(14,6))    
def get_heat_map(mat, x, y, color):
##    print(mat)
#    fig = matplotlib.figure.Figure(figsize=(14,8))
#    ax = fig.add_subplot(1, 1, 1) 
#    
##    print("adddd")
#    im, cbar = heatmap(mat, x, y, ax=ax,
#                       cmap=color, cbarlabel="harvest [t/year]")
#    texts = annotate_heatmap(im, valfmt="{x:.2f}")
##    print("here")
#    fig.tight_layout()
    print("jjfjfj")
    i+=1
    fig, ax = plt.subplots(figsize=(14,6), num=i%2, clear=True)
    print("fin")
    
#    fig, ax = plt.subplots(figsize=(14,6), num=0)
    if color == "RdYlGn":     # Sample figsize in inches
        sns.heatmap(mat,annot=True, fmt='.2f', cmap=color, center = 0, xticklabels=y, yticklabels=x, ax=ax) 
    else:
        sns.heatmap(mat,annot=True, fmt='.2f', cmap=color, vmin = 0, xticklabels=y, yticklabels=x, ax=ax )
    
    print("plt close")
    plt.close()
    print("plt close")
    return fig

"""

def plot_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):
    ''' 
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor
    
    Returns:
        summary: TensorFlow summary 
    
    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc. 
        - Currently, some of the ticks dont line up due to rotations.
    '''
    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')
    
    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()
    
    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')
    
    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]
    
    tick_marks = np.arange(len(classes))
    
    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()
    
    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    return summary
"""
''' confusion matrix summaries '''
img_d_summary_dir = os.path.join(checkpoint_dir, "summaries", "img")
img_d_summary_writer = tf.summary.FileWriter(img_d_summary_dir, sess.graph)
img_d_summary = plot_confusion_matrix(correct_labels, predict_labels, labels, tensor_name='dev/cm')
img_d_summary_writer.add_summary(img_d_summary, current_step)

"""