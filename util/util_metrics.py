import csv
import sklearn.metrics as SKM
import matplotlib.pyplot as plt
import numpy as np

from . import util_main as UMN
from . import util_probing as UP
from . import util_constants as UC


# take an array of dicts and turn it in to a dict of arrays
def accum_metrics_to_metric_dict(accum_metrics):
    ret = {}
    for i in range(len(accum_metrics)):
        first_time = False
        if i == 0:
            first_time = True
        for k,v in accum_metrics[i].items():
            # don't save confusion matric
            if k != 'cm':
                if first_time == True:
                    ret[k] = []
                ret[k].append(v)
    return ret

def save_results_to_csv(resdict, ):
    res_path = None
    other_str = get_save_other_str(preq_idx, layer_idx)
    res_path = 'res'
    save_path = UMN.get_save_path(res_path, configdict, other=other_str, make_dir = True) 
    cur_header = list(resdict.keys())
    f = open(save_path, 'w')
    csvw = csv.DictWriter(f, fieldnames=cur_header)
    csvw.writeheader()
    csvw.writerow(resdict)
    f.close()

def make_confusion_matrix(truths, preds):
    figsize = None
    hide_labels = True
    figsize = UC.CM_FIGSIZE_S
        
    # plot confusion matrix
    fig, ax = plt.subplots(figsize=figsize)
    # convert indices to class strings before feeding into confusion matrix
    cmd = SKM.ConfusionMatrixDisplay.from_predictions(
            [datadict['idxdict'][x] for x in truths],
            [datadict['idxdict'][x] for x in preds],
            labels=datadict['label_arr'],
            normalize='true',
            include_values=hide_labels == False,
            cmap="Purples",
            colorbar=hide_labels == True,
            )

    if hide_labels == True:
        tick_positions = np.arange(0, datadict['num_classes'], 5)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions, fontsize=9)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_positions, fontsize=9)

    expr_name = UC.EXPR_PRETTY_NAMES[configdict['expr_type']]
    dataset_name = UC.DATASET_PPRINT[configdict['dataset']]
    cur_train_prop = get_preq_train_prop(preq_idx, configdict)
    cur_train_pct = f'{cur_train_prop * 100:.2f}%'
    title = f'{dataset_name} ({cur_train_pct} of Train Data) {expr_name} Results'
    ax.set_title(title)
    fig.tight_layout()
    other_str = get_save_other_str(preq_idx, layer_idx)
    res_path = 'cm'
    save_path = UMN.get_save_path(res_path, configdict, other=other_str, make_dir = True) 
    plt.savefig(save_path)
    plt.clf()
    return cmd.confusion_matrix


def calculate_log2_nll_sum(nll_arr):
    return np.sum(nll_arr) * UC.LOG2E

# make_cm = make confusion matrix
def get_classification_metrics(truths, preds, layer_idx, trial_number, save_to_csv = False, make_cm = False):

    ret = {} 
    ret['layer_idx'] = layer_idx
    ret['accuracy_score']= SKM.accuracy_score(truths, preds)
    ret['f1_macro'] = SKM.f1_score(truths, preds, average='macro')
    ret['f1_micro'] = SKM.f1_score(truths, preds, average='micro')
    ret['balanced_accuracy_score'] = SKM.balanced_accuracy_score(truths, preds)

    # only save for eval
    if save_to_csv == True:
        save_results_to_csv(ret, configdict, preq_idx, layer_idx)
    if make_cm == True:
        ret['cm'] = make_confusion_matrix(truths, preds, preq_idx, layer_idx, datadict, configdict)
    return ret


def get_optimization_metric(metric_dict, is_balanced = True):
    ret = None
    if is_balanced == True:
        ret = metric_dict['accuracy_score']
    else:
        ret = metric_dict['balanced_accuracy_score']

    return ret

