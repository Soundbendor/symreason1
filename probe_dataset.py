import torch, torch.utils.data as TUD
import os

import util.util_data as UD
import util.util_probing as UP

import util.util_chords as CH

# triple_token_idx selects the index of the note triple to pick out (-1 selects all)
class ProbeDataset(TUD.Dataset):
    def __init__(self, data_df=None, csv_dfs=None, layer_idx=0, from_dir = '', triple_token_idx = -1, by_category = False, device ='cpu'):
        self.data_df = data_df
        self.csv_dfs = csv_dfs
        self.by_category = by_category
        self.layer_idx = layer_idx
        self.device = device
        self.from_dir = from_dir
        #self.to_torch = to_torch
        self.triple_token_idx = triple_token_idx

    def __len__(self):
        return self.data_df['basename'].count()

    def set_layer_idx(self,idx):
        self.layer_idx = idx

    def __getitem__(self,idx):
        cur_row = self.data_df.row(idx, named=True)
        cur_basename = cur_row['basename']
        cur_csv = self.csv_dfs[cur_row['csv_basename']]
        cur_truth = None
        if self.by_category == False:
            cur_truth = torch.tensor(cur_csv['chord_scaledeg'])
        else:
            cur_truth = torch.tensor(cur_csv['chord_cat_int'])
        # hidden state, tokenized, ...
        cur_arr = UP.filter_hidden(
                torch.load(os.path.join(self.from_dir, f'{cur_basename}-hidden.pt')),
                torch.load(os.path.join(self.from_dir, f'{cur_basename}-tokens.pt')),
                triple_token_idx = self.triple_token_idx,
                filt_tokens = UP.FILT_TOKENS
                ).to(self.device)
        # should come out (layers, seqlen, hidden_dim)
        # so select by layer_idx
        return cur_arr[self.layer_idx], cur_truth




