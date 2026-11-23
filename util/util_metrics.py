import csv, os
import sklearn.metrics as SKM
import matplotlib.pyplot as plt
import numpy as np

from . import util_probing as UP
from . import util_chords as CH

CONFMAT_FOLDER = 'confmat'
RES_CSV_FOLDER = 'res_csv'
# take an array of dicts and turn it in to a dict of arrays
def accum_metrics_to_metric_dict(accum_metrics):
    ret = {}
    for i in range(len(accum_metrics)):
        first_time = False
        if i == 0:
            first_time = True
        for k,v in accum_metrics[i].items():
            # don't save confusion matrix
            if k != 'cm':
                if first_time == True:
                    ret[k] = []
                ret[k].append(v)
    return ret

def save_results_to_csv(resdict, save_name):
    save_path = os.path.join(RES_CSV_FOLDER, f'{save_name}.csv')
    cur_header = list(resdict.keys())
    f = open(save_path, 'w')
    csvw = csv.DictWriter(f, fieldnames=cur_header)
    csvw.writeheader()
    csvw.writerow(resdict)
    f.close()

def make_confusion_matrix(truths, preds, by_category = False, param_str = '', save_name = ''):
    figsize = None
    hide_labels = True
    figsize = UC.CM_FIGSIZE_S
    idx_dict = None
    label_arr = None
    expr_type = None
    if by_category == False:
        idx_dict = CH.idx_to_roman
        label_arr = CH.generic_roman_arr
        expr_type = 'Roman Numeral'
    else:
        idx_dict = CH.idx_to_chord_cat
        label_arr = CH.chord_cat_arr
        expr_type = 'Chord Category'
    num_classes = len(label_arr)
    # plot confusion matrix
    fig, ax = plt.subplots(figsize=figsize)
    # convert indices to class strings before feeding into confusion matrix
    cmd = SKM.ConfusionMatrixDisplay.from_predictions(
            [idx_dict[x] for x in truths],
            [idx_dict[x] for x in preds],
            labels=label_arr,
            normalize='true',
            include_values=hide_labels == False,
            cmap="Purples",
            colorbar=hide_labels == True,
            )

    if hide_labels == True:
        tick_positions = np.arange(0, num_classes, 5)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions, fontsize=9)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_positions, fontsize=9)

    title = f'{expr_type} Results ({param_str})'
    ax.set_title(title)
    fig.tight_layout()
    save_path = os.path.join(CONFMAT_FOLDER, f'{save_name}.png')
    plt.savefig(save_path)
    plt.clf()
    return cmd.confusion_matrix



# make_cm = make confusion matrix
def get_classification_metrics(truths, preds, layer_idx = 0, by_category = False, param_str = '', save_name = '', make_cm = False):

    ret = {} 
    ret['layer_idx'] = layer_idx
    ret['accuracy_score']= SKM.accuracy_score(truths, preds)
    ret['f1_macro'] = SKM.f1_score(truths, preds, average='macro')
    ret['f1_micro'] = SKM.f1_score(truths, preds, average='micro')
    ret['balanced_accuracy_score'] = SKM.balanced_accuracy_score(truths, preds)

    # only save for eval
    if len(save_name) > 0:
        save_results_to_csv(ret, save_name)
    if make_cm == True:
        ret['cm'] = make_confusion_matrix(truths, preds, by_category = by_category, param_str = param_str, save_name = save_name)
    return ret


def get_optimization_metric(metric_dict, is_balanced = True):
    ret = None
    if is_balanced == True:
        ret = metric_dict['accuracy_score']
    else:
        ret = metric_dict['balanced_accuracy_score']

    return ret

