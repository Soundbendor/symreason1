import torch
import numpy as np

# 0 = start, 1 = end, 5 = <T>, 6-38 = prefixes
FILT_TOKENS = torch.concat((torch.tensor([0,1,5]), torch.arange(6,39)))

def get_token_translator_dict(tokenizer):
    return {int(v):k for (k,v) in tokenizer.get_vocab().items()}

def get_token_idxs(tokenized, filt_tokens = FILT_TOKENS):
    return torch.where(torch.isin(tokenized, filt_tokens, invert = True))[0]

def reshape_as_triple(filt_tokenized):
    return filt_tokenized.view(*filt_tokenized.shape[:-2], filt_tokenized.shape[-2]//3, filt_tokenized.shape[-1] * 3)
               
# 2 = last position of triple
def get_idx_of_triple(filt_tokenized,idx=2):
    # ndim = 4: num_layers, batch_size, seqlen, dim
    # ndim = 3: batch_size, seqlen, dim
    # ndim = 2: seqlen, dim
    return filt_tokenized[...,idx::3, :]


def filter_by_token_idxs(hidden, token_idxs):
    # ndim = 4: num_layers, batch_size, seqlen, dim
    # ndim = 3: batch_size, seqlen, dim
    # ndim = 2: seqlen, dim
    return hidden[...,token_idxs, :]


# triple_token_idx selects the index of the note triple to pick out (-1 selects all)
def filter_hidden(hidden, tokenized, triple_token_idx = -1,  filt_tokens = FILT_TOKENS):
    cur_filt = filter_by_token_idxs(
            hidden,
            get_token_idxs(tokenized, filt_tokens = filt_tokens))
    if triple_token_idx < 0:
        return reshape_as_triple(cur_filt)
    else:
        return get_idx_of_triple(cur_filt)

# input torch, output numpy
# predictions are probability dists, convert to index
def accumulate_truths_preds(truths, truths_to_add, preds, preds_to_add, batch_idx):
    new_truths = truths_to_add.detach().cpu().numpy().flatten()
    new_preds = torch.argmax(preds_to_add,axis=1).detach().cpu().numpy().flatten()

    # first time through, just return new truths and preds
    if batch_idx == 0:
        return new_truths, new_preds
    else:
        # The base of an array that owns its memory is None
        # (and want to own own memory, so deep copy if not)
        # (doesn't work if truths is None, first time around)
        if truths.base is None and preds.base is None:
            return np.hstack((truths,new_truths)), np.hstack((preds, new_preds))
        else:
            return np.hstack((copy.deepcopy(truths),new_truths)), np.hstack((copy.deepcopy(preds), new_preds))

