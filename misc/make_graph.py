import polars as pl
import os
import matplotlib.pyplot as plt
CSVDIR = 'res_csv'
RESGRAPH_DIR = 'res_graph'
keys = ['layer_idx','accuracy_score','f1_macro','f1_micro','balanced_accuracy_score']


csvs = ['l_0-lrexp_-3-wdexp_-3-bs_512-tti_-1-bc_0-trial_22-1.csv', 'l_10-lrexp_-3-wdexp_-3-bs_512-tti_-1-bc_0-trial_42-1.csv', 'l_11-lrexp_-2-wdexp_-3-bs_512-tti_-1-bc_0-trial_2-1.csv', 'l_12-lrexp_-2-wdexp_-3-bs_512-tti_-1-bc_0-trial_5-1.csv', 'l_13-lrexp_-2-wdexp_-3-bs_512-tti_-1-bc_0-trial_16-1.csv', 'l_14-lrexp_-2-wdexp_-2-bs_512-tti_-1-bc_0-trial_54-1.csv', 'l_15-lrexp_-2-wdexp_-2-bs_512-tti_-1-bc_0-trial_43-1.csv', 'l_16-lrexp_-2-wdexp_-2-bs_512-tti_-1-bc_0-trial_65-1.csv', 'l_1-lrexp_-3-wdexp_-2-bs_512-tti_-1-bc_0-trial_63-1.csv', 'l_2-lrexp_-3-wdexp_-2-bs_512-tti_-1-bc_0-trial_8-1.csv', 'l_3-lrexp_-3-wdexp_-3-bs_512-tti_-1-bc_0-trial_12-1.csv', 'l_4-lrexp_-3-wdexp_-3-bs_512-tti_-1-bc_0-trial_18-1.csv', 'l_5-lrexp_-3-wdexp_-3-bs_512-tti_-1-bc_0-trial_38-1.csv', 'l_6-lrexp_-3-wdexp_-3-bs_512-tti_-1-bc_0-trial_15-1.csv', 'l_7-lrexp_-3-wdexp_-3-bs_512-tti_-1-bc_0-trial_57-1.csv', 'l_8-lrexp_-3-wdexp_-3-bs_512-tti_-1-bc_0-trial_62-1.csv', 'l_9-lrexp_-3-wdexp_-3-bs_512-tti_-1-bc_0-trial_1-1.csv']

res = {k: [0.] * 17 for k in keys}
for layer_idx in range(17):
    for cur in csvs:
        found = False
        df = None
        if f'l_{layer_idx}-' in cur[:5]:
            print(f'l_{layer_idx}')
            found = True
            cur_f = os.path.join(CSVDIR, cur)
            df = pl.read_csv(cur_f)
        if found == True:
            for k in keys:
                res[k][layer_idx] = df[k][0]

def plot_per_metric(cur_key, cur_y):
    cur_title = f'{cur_key} Across Layer Indices'
    fname = f'{cur_key}-roman.png'
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(list(range(17)), cur_y)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel(cur_key)
    ax.set_title(cur_title)
    plt.tight_layout()
    
    fpath = os.path.join(RESGRAPH_DIR, fname)
    plt.savefig(fpath)
    fig.clear()
    plt.close()

for k in keys:
    plot_per_metric(k, res[k])
