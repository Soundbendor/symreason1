import torch, torch.utils.data as TUD
import os

import util.util_data as UD
import util.util_main as UMN

import chords as CH
class ProbeDataset(TUD.Dataset):
    def __init__(self, data_df=None, csv_dfs=None, layer_idx=0, from_dir = '', by_category = False, to_torch = True, device ='cpu'):
        self.data_df = data_df
        self.csv_dfs = csv_dfs
        self.by_category = by_category
        self.layer_idx = layer_idx
        self.device = device
        self.from_dir = from_dir
        self.to_torch = to_torch

    def __len__(self):
        return self.data_df['fname'].count()

    def set_layer_idx(self,idx):
        self.layer_idx = idx

    def __getitem__(self,idx):
        cur_row = self.df.row(idx, named=True)
        cur_fname = cur_row['fname']
        cur_csv = self.csv_dfs[cur_row['csv_basename']]
        cur_truth = None
        if self.by_category == False:
            cur_truth = torch.tensor(cur_csv['chord_scaledeg'])
        else:
            cur_truth = torch.tensor(cur_csv['chord_cat_int'])
        cur_arr = torch.load(os.path.join(self.from_dir, cur_fname))
        return cur_arr, cur_truth




