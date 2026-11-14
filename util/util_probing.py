import torch

# 0 = start, 1 = end, 5 = <T>, 6-38 = prefixes
FILT_TOKENS = torch.concat((torch.tensor([0,1,5]), torch.arange(6,39)))

def get_token_translator_dict(tokenizer):
    return {int(v):k for (k,v) in tokenizer.get_vocab().items()}

def get_token_idxs(tokenized, filt_tokens = FILT_TOKENS):
    return torch.where(tokenized, filt_tokens, invert = True)[0]

def reshape_as_triple(filt_tokenized):
    return filt_tokenized.view(*filt_tokenized[:-2], filt_tokenized[-2]//3, filt_tokenized[-1] * 3)
               
# 2 = last position of triple
def get_idx_of_triple(filt_tokenized,idx=2):
    if filt_tokenized.ndim == 4:
        # num_layers, batch_size, seqlen, dim
        return filt_tokenized[:,:,[idx::3],:]
    elif filt_tokenized.ndim == 3:
        # batch_size, seqlen, dim
        return filt_tokenized[:,[idx::3],:]
    elif filt_tokenized.ndim == 2:
        # seqlen, dim
        return filt_tokenized[[idx::3], :]



