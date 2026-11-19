from collections import OrderedDict
from torch import nn
import os
import torch

class LinearProbe(nn.Module):
    def __init__(self, in_dim =1536, hidden_dims = [], out_dim = 10):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_hidden = len(hidden_dims)
        self.num_layers = self.num_hidden + 1

        cur_layers = []

        prev_dim = in_dim
        hidden_idx = 0
        relu_idx = 0
        for hidden_dim in hidden_dims:
            cur_layers.append( (f'linear_{hidden_idx}', nn.Linear(prev_dim, hidden_dim, bias = True)) )
            cur_layers.append( (f'relu_{relu_idx}', nn.ReLU()) )
            prev_dim = hidden_dim
            hidden_idx += 1
            relu_idx += 1

        cur_layers.append( (f'linear_{hidden_idx}', nn.Linear(prev_dim, out_dim, bias = True)) )
        self.layers = nn.Sequential(OrderedDict(cur_layers))

    def forward(self, x):
        out = self.layers(x)
        return out

def init_probe(by_category = False, trpile_token_idx = -1, hidden_dims = []):
    out_dim = 3
    if by_category == False:
        out_dim = 7
    in_dim = 1536
    if triple_token_idx < 0:
        in_dim *= 3
    return LinearProbe(in_dim =in_dim, hidden_dims = hidden_dims, out_dim = out_dim)

