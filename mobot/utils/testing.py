import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

from .vis import *

def read_scores(test_file):
    
    # Return a dict contains the scores of each model {key(model):value(scores)}
    # test_file: the path stored test result or the test result dict

    if isinstance(test_file,str):
        print("Reading test results in",test_file)
        dict = np.load(test_file,allow_pickle = True).item()
    else:
        dict = test_file
        
    models = list(dict.keys())
    scores = list(dict.values())

    model_APs = OrderedDict()

    for model,score in zip(models,scores):
        model_APs[model] = {}
        model_APs[model]['bbox-AP-end'] = score['bbox']['AP-end']
        model_APs[model]['bbox-AP-side'] = score['bbox']['AP-side']
        model_APs[model]['segm-AP-end'] = score['segm']['AP-end']
        model_APs[model]['segm-AP-side'] = score['segm']['AP-side']

    return model_APs


def plot_test(model_APs):

    models = model_APs.keys()
    x_axis = list(map(lambda x:x.split('_')[-1][-5:],models))

    segm_AP_ends = list(map(lambda x:x['segm-AP-end'],model_APs.values()))
    segm_AP_sides = list(map(lambda x:x['segm-AP-side'],model_APs.values()))
    bbox_AP_ends = list(map(lambda x:x['bbox-AP-end'],model_APs.values()))
    bbox_AP_sides = list(map(lambda x:x['bbox-AP-side'],model_APs.values()))

    plt.figure(figsize=figsize)
    plt.tick_params(labelsize=labelsize)

    plt.plot(x_axis,segm_AP_ends,label='segm_AP_ends',marker='.',c=c_segm_AP_ends,linewidth=linewidth)
    plt.plot(x_axis,segm_AP_sides,label='segm_AP_sides',marker='.',c=c_segm_AP_sides,linewidth=linewidth)
    plt.plot(x_axis,bbox_AP_ends,label='bbox_AP_ends',marker='.',c=c_bbox_AP_ends,linewidth=linewidth)
    plt.plot(x_axis,bbox_AP_sides,label='bbox_AP_sides',marker='.',c=c_bbox_AP_sides,linewidth=linewidth)

    ind = np.argmax(segm_AP_ends)
    print('According to segm_AP_ends, best model is',list(models)[ind])

    plt.vlines(x_axis[ind],plt.ylim()[0], plt.ylim()[1],linestyles='dashed',color=c_vline,linewidth=linewidth)
    np.argmax(segm_AP_ends)
    plt.rcParams.update({'font.size':legendsize})
    plt.legend()
    plt.show()